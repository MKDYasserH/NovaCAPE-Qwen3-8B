"""Fit one benchmark-independent post head on terminal steps, Q&A and multi-view preferences."""
import hashlib
import json
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from finetuning.chunks import ChunkStore
from finetuning.environment import app, config, volume
from finetuning.heads import ContrastiveHeads
from finetuning.loss import contrastive_loss, embed_batch
from finetuning.post_sweep import ROOT, RESOURCES, assess, replay_sample, terminal_pairs
from schemas.training import ChunkId

PREFERENCE_STEM = "general-preferences-40b436560ca83a8dba36114c22ab3c66e43f6d5e"
AGENT_STEM = "general-agents-eb95b66ca30ef071d5e49495d5d780c29f9aef7b"
SEED = 1234
CHECKS = (40, 80, 160, 240, 320)


def margins(model, vectors, indices):
    """Compute paired score differences for two answers and the judges' prose about them."""
    states = model.embed_states(vectors[indices[:, 0]])
    choices = model.embed_actions(vectors[indices[:, 1:]])
    logits = model.logit_scale() * (states[:, None] * choices).sum(-1)
    return logits[:, 0] - logits[:, 1], logits[:, 2] - logits[:, 3]


def preference_loss(answer, review, use_reviews):
    if not use_reviews:
        return F.softplus(-answer).mean()
    # The critique supplies a second supervised view and a soft target for the answer-only view.
    teacher = review.detach().sigmoid()
    return (0.45 * F.softplus(-answer).mean() + 0.45 * F.softplus(-review).mean()
            + 0.10 * F.binary_cross_entropy_with_logits(answer, teacher))


def read_cache(stem, device):
    folder = ROOT / "post_inputs"
    source = (folder / (stem + ".json")).read_bytes()
    manifest = json.loads((folder / (stem + ".manifest.json")).read_text())
    if manifest["sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("Source and frozen embedding manifest disagree")
    rows = json.loads(source)["rows"]
    with np.load(folder / (stem + ".npz")) as cached:
        vectors = torch.tensor(cached["vectors"], dtype=torch.float32, device=device)
        indices = cached["indices"]
        if not np.array_equal(cached["train"], [r["split"] == "train" for r in rows]):
            raise ValueError("Cached split disagrees with source")
    return manifest, rows, vectors, indices


def domain_choices(rows, indices, split):
    """Other-goal actions from the same environment give seven distinct, plausible distractors."""
    rng = np.random.default_rng(SEED)
    groups = {}
    for i, row in enumerate(rows):
        if row["split"] == split:
            groups.setdefault(row["source"], []).append(i)
    by_domain = {}
    for domain, members in groups.items():
        choices = []
        for i in members:
            pool = np.unique([indices[j, 1] for j in members if rows[j]["group"] != rows[i]["group"]
                              and indices[j, 1] != indices[i, 1]])
            if len(pool) < 7:
                raise ValueError(f"Too few distinct alternatives in {domain}")
            for _ in range(4 if split == "train" else 1):
                choices.append([*indices[i], *rng.choice(pool, 7, replace=False)])
        by_domain[domain] = np.array(choices)
    return by_domain


@torch.no_grad()
def mine_domain_choices(model, vectors, rows, pairs):
    """Find confusable other-goal actions, excluding every known action of this goal."""
    rng = np.random.default_rng(SEED + 1)
    domains = sorted({row["source"] for row in rows})
    result = {}
    for domain in domains:
        members = [i for i, r in enumerate(rows) if r["split"] == "train" and r["source"] == domain]
        subset = pairs[members]
        action_ids = np.unique(subset[:, 1])
        candidates = model.embed_actions(vectors[action_ids])
        queries = model.embed_states(vectors[subset[:, 0]])
        similarity = (queries @ candidates.T).cpu().numpy()
        resemblance = (vectors[subset[:, 1]] @ vectors[action_ids].T).cpu().numpy()
        known = {}
        for i in members:
            known.setdefault(rows[i]["group"], set()).add(pairs[i, 1])
        choices = []
        for j, i in enumerate(members):
            outside_goal = np.flatnonzero(~np.isin(action_ids, list(known[rows[i]["group"]])))
            plausible = outside_goal[resemblance[j, outside_goal] < 0.95]
            if len(plausible) < 7:
                # Some environments use near-identical templates. Keep the most distinct alternatives.
                plausible = outside_goal[np.argsort(resemblance[j, outside_goal])[:7]]
            if len(plausible) < 7:
                continue
            hardest = plausible[np.argsort(-similarity[j, plausible])[:16]]
            for _ in range(4):
                hard = rng.choice(hardest, 3, replace=False)
                random = rng.choice(np.setdiff1d(plausible, hard), 4, replace=False)
                choices.append([*pairs[i], *action_ids[hard], *action_ids[random]])
        if not choices:
            raise ValueError(f"No eligible mined examples in {domain}")
        result[domain] = np.array(choices)
    return result


def choice_logits(model, vectors, rows):
    states = model.embed_states(vectors[rows[:, 0]])
    options = model.embed_actions(vectors[rows[:, 1:]])
    return model.logit_scale() * (states[:, None] * options).sum(-1)


@torch.no_grad()
def evaluate(model, vectors, dev_rows, agents, qa, actions, choices):
    metrics = assess(model, agents, qa, vectors.device)
    answer, review = zip(*(margins(model, vectors, rows) for rows in dev_rows.split(128)))
    metrics.update(answer_preference=(torch.cat(answer) > 0).float().mean().item(),
                   review_preference=(torch.cat(review) > 0).float().mean().item())
    metrics["action_by_domain"] = {
        domain: torch.cat([choice_logits(model, actions, batch).argmax(1) == 0
                           for batch in rows.split(128)]).float().mean().item()
        for domain, rows in choices.items()
    }
    metrics["action_macro"] = float(np.mean(list(metrics["action_by_domain"].values())))
    return metrics


def selection_score(metrics, baseline):
    # Equal weight for the four data families. Normalize each gain by remaining errors.
    def error_reduction(metric):
        return (metrics[metric] - baseline[metric]) / max(1e-6, 1 - baseline[metric])
    preference = (2 * error_reduction("answer_preference") + error_reduction("review_preference")) / 3
    return (preference + error_reduction("agent_top1") + error_reduction("qa_top1") + error_reduction("action_macro")) / 4


@app.function(**{**RESOURCES, "memory": 24576}, gpu=["L4", "A10G", "L40S"], timeout=1200, max_containers=1)
def fit(parent: str = ""):
    volume.reload()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)
    manifest, pref_meta, vectors, pref_idx = read_cache(PREFERENCE_STEM, device)
    indices = torch.as_tensor(pref_idx, device=device)
    train_mask = torch.tensor([r["split"] == "train" for r in pref_meta], device=device)
    train_rows, dev_rows = indices[train_mask], indices[~train_mask]
    agent_manifest, rows, actions, pairs = read_cache(AGENT_STEM, device)
    for key in ("group", "trajectory"):
        groups = [{(r["source"], r[key]) for r in rows if r["split"] == split} for split in ("train", "dev")]
        if groups[0] & groups[1]:
            raise ValueError(f"ADP {key} occurs on both sides of the split")
    train_choices = domain_choices(rows, pairs, "train")
    dev_choices = {key: torch.as_tensor(value, device=device)
                   for key, value in domain_choices(rows, pairs, "dev").items()}
    train, _, train_ids = terminal_pairs("train")
    dev, _, dev_ids = terminal_pairs("val")
    if train_ids & dev_ids:
        raise ValueError("Terminal training and validation tasks overlap")
    rng = np.random.default_rng(SEED)
    validation = dev.take(rng.choice(dev.num_rows, size=4096, replace=False))
    chunks = [ChunkId(shard="shard_00", index=i) for i in (0, 7, 14, 21)]
    if config.held_out.all_chunks.intersection(chunks):
        raise ValueError("Q&A training or development overlaps final evaluation")
    store = ChunkStore(config.data_dir)
    replay = replay_sample(store, chunks[:-1], rng, 8192)
    guard = replay_sample(store, chunks[-1:], rng, 4096)
    out = ROOT / config.runs_dir / f"post-general-{uuid4().hex[:10]}"
    out.mkdir()
    mid = torch.load(ROOT / config.runs_dir / "mid/best.pt", map_location="cpu", weights_only=True)
    model = ContrastiveHeads(config.head, config.loss).to(device)
    model.load_checkpoint_weights(mid)
    model.log_logit_scale.requires_grad_(False)
    baseline = evaluate(model, vectors, dev_rows, validation, guard, actions, dev_choices)
    starting = mid
    if parent:
        # All candidate generation and selection below uses only pre-defined development splits.
        folder = ROOT / config.runs_dir / parent
        parent_report = json.loads((folder / "selection.json").read_text())
        if (parent_report.get("parent") != "mid" or parent_report["seed"] != SEED
                or parent_report["preference_data"] != manifest or parent_report["adp_data"] != agent_manifest):
            raise ValueError("Continuation requires a matching general first-round checkpoint")
        starting = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model.load_checkpoint_weights(starting)
    history = []
    selected = dict(recipe="mid", step=0, score=0.0, **baseline)
    cfg = dict(width=config.head.width, depth=3, projection_dim=config.head.projection_dim,
               activation="gelu", layernorm=True, residual=False, hidden_size=config.head.input_dim)

    def save():
        torch.save({**model.checkpoint_weights(), "cfg": cfg, "selection": selected}, out / "best.partial")
        (out / "best.partial").replace(out / "best.pt")

    def consider(recipe, step):
        nonlocal selected
        metrics = evaluate(model, vectors, dev_rows, validation, guard, actions, dev_choices)
        entry = dict(recipe=recipe, step=step, score=selection_score(metrics, baseline), **metrics)
        history.append(entry)
        print(entry, flush=True)
        if (metrics["qa_top1"] >= baseline["qa_top1"] - 0.005 and
                metrics["agent_top1"] >= baseline["agent_top1"] and entry["score"] > selected["score"]):
            selected = entry
            save()

    def sample_loss(pool, size, rng):
        rows = rng.choice(pool.num_rows, size, replace=False)
        return contrastive_loss(embed_batch(model, pool.take(rows).to_device(device)))

    # Both fits share architecture, initialization and batches. Average a fixed late window.
    average = {name: torch.zeros_like(value) for name, value in model.state_dict().items()}
    count = 0
    save()
    recipes = ("answer_only", "answer_and_review")
    mined = {}
    if parent:
        consider("starting_general", 0)
        mined = mine_domain_choices(model, actions, rows, pairs)
        recipes = ("continued_random", "continued_mined")
    for recipe in recipes:
        model.load_checkpoint_weights(starting)
        rng = np.random.default_rng(SEED + int(bool(parent)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00004 if parent else 0.00008, weight_decay=0)
        for step in range(1, CHECKS[-1] + 1):
            agent_loss = sample_loss(train, 256, rng)
            qa_loss = sample_loss(replay, 128, rng)
            pref_rows = train_rows[rng.choice(len(train_rows), 64, replace=False)]
            answer, review = margins(model, vectors, pref_rows)
            distractors = mined if recipe == "continued_mined" and step % 2 else train_choices
            sampled = np.concatenate([choices[rng.integers(len(choices), size=16)]
                                      for choices in distractors.values()])
            logits = choice_logits(model, actions, torch.as_tensor(sampled, device=device))
            domain_loss = F.cross_entropy(logits, torch.zeros(len(logits), device=device, dtype=torch.long))
            loss = (0.25 * agent_loss + 0.25 * qa_loss + 0.20 * domain_loss +
                    0.30 * preference_loss(answer, review, recipe != "answer_only"))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in CHECKS:
                consider(recipe, step)
                if step >= 160:
                    for name, value in model.state_dict().items():
                        average[name].add_(value)
                    count += 1
    model.load_state_dict({name: value / count for name, value in average.items()})
    consider("mean_of_both_continuations" if parent else "mean_of_both_recipes_at_160_240_320", 320)
    report = dict(run=out.name, parent=parent or "mid", mid_step=mid["step"], seed=SEED, preference_data=manifest,
                  adp_data=agent_manifest, adp_train={k: len(v) // 4 for k, v in train_choices.items()},
                  adp_dev={k: len(v) for k, v in dev_choices.items()}, preference_train=len(train_rows), preference_dev=len(dev_rows), terminal_train=train.num_rows,
                  terminal_dev=validation.num_rows, qa_train=replay.num_rows, baseline=baseline,
                  selected=selected, history=history)
    (out / "selection.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    print(f"Locked checkpoint {out.name}", flush=True)
    return out.name
