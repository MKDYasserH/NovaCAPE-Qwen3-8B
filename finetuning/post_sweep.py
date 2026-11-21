"""Budgeted post-training on cached terminal-agent and hard-negative Q&A embeddings."""
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from finetuning.chunks import ChunkStore
from finetuning.environment import app, config, image, volume
from finetuning.heads import ContrastiveHeads
from finetuning.loss import contrastive_loss, count_top1_hits, embed_batch, true_vs_hard_negative_logits
from schemas.training import ChunkId, PairArrays

DATASET = "Hkang/litecoder-terminal-sft-qwen3-8b-embeddings"
REVISION = "5045b1d5f673419300f7ce573542719ac49cecc2"
ROOT = Path(config.data_dir)
CACHE = ROOT / "post_inputs" / REVISION
RESOURCES = dict(image=image, volumes={config.data_dir: volume}, cpu=4, memory=16384, retries=0)
SEED = config.seed
AGENT_BATCH = 256
QA_BATCH = 128
STEPS = 120
CHECK_EVERY = 40
QA_TOLERANCE = 0.015
LEARNING_RATE = 0.0001


@app.function(**RESOURCES, timeout=900)
def prepare():
    volume.reload()
    snapshot_download(DATASET, repo_type="dataset", revision=REVISION, local_dir=CACHE,
                      allow_patterns=["train/*.pt", "train/metadata.json", "val/*.pt", "val/metadata.json"])
    volume.commit()
    print(f"Terminal-agent inputs ready at {CACHE}", flush=True)


def terminal_pairs(split: str):
    folder = CACHE / split
    samples = json.loads((folder / "metadata.json").read_text())["samples"]
    tasks = [s["task_id"] for s in samples]
    if any(not t for t in tasks):
        raise ValueError(f"Missing task IDs in {split}")
    _, task_ids = np.unique(tasks, return_inverse=True)
    trajectories = {}
    for row, sample in enumerate(samples):
        trajectories.setdefault((tasks[row], sample["trajectory_id"]), []).append((sample["step_idx"], row))
    partner = np.full(len(samples), -1, dtype=np.int64)
    for steps in trajectories.values():
        steps.sort()
        for i, (step, row) in enumerate(steps):
            other_step, other = steps[(i + len(steps) // 2) % len(steps)]
            if abs(step - other_step) >= 3:
                partner[row] = other

    def matrix(name):
        tensor = torch.load(folder / f"{name}_embeddings.pt", map_location="cpu", weights_only=True)
        if tensor.shape != (len(samples), config.head.input_dim):
            raise ValueError(f"{split}/{name} shape {tensor.shape} disagrees with metadata")
        return tensor.numpy()

    pairs = PairArrays(states=matrix("state"), actions=matrix("action"), doc_ids=task_ids)
    return pairs, partner, set(tasks)


def replay_sample(store, chunks, rng, rows_per_chunk):
    parts = []
    for chunk in chunks:
        pairs = store.read(chunk, with_hard_negatives=True, mmap_mode="r")
        # Contiguous reads avoid thousands of random reads from a remote volume.
        n = min(rows_per_chunk, pairs.num_rows)
        start = rng.integers(pairs.num_rows - n + 1)
        parts.append(pairs.take(np.arange(start, start + n)))
    return PairArrays.concat(parts)


def swap_loss(model, pairs, partners, rows, embedded, device):
    """Prefer the observed assignment of two distant steps to swapping their actions."""
    others = partners[rows]
    valid = others >= 0
    idx = np.flatnonzero(valid)
    if not len(idx):
        return embedded.states.new_zeros(())
    alternate = pairs.take(others[valid]).to_device(device)
    # Repeated or paraphrased commands provide ambiguous supervision.
    original_actions = torch.from_numpy(pairs.actions[rows[valid]]).to(device).float()
    distinct = F.cosine_similarity(original_actions, alternate.actions) < 0.90
    idx = torch.as_tensor(idx, device=device)
    state_diff = embedded.states[idx] - model.embed_states(alternate.states)
    action_diff = embedded.actions[idx] - model.embed_actions(alternate.actions)
    advantage = embedded.scale * (state_diff * action_diff).sum(dim=-1) / 2
    return (F.softplus(-advantage) * distinct).sum() / distinct.sum().clamp_min(1)


@torch.no_grad()
def assess(model, agents, qa, device):
    hits = 0
    for start in range(0, agents.num_rows, AGENT_BATCH):
        rows = np.arange(start, min(start + AGENT_BATCH, agents.num_rows))
        hits += count_top1_hits(embed_batch(model, agents.take(rows).to_device(device)))
    qa_hits = qa_total = 0
    for start in range(0, qa.num_rows, QA_BATCH):
        rows = np.arange(start, min(start + QA_BATCH, qa.num_rows))
        emb = embed_batch(model, qa.take(rows).to_device(device))
        logits = true_vs_hard_negative_logits(emb)[emb.negative_counts > 0]
        qa_hits += (logits.argmax(dim=1) == 0).sum().item()
        qa_total += len(logits)
    if not qa_total:
        raise ValueError("Retention sample has no valid hard negatives")
    return dict(agent_top1=hits / agents.num_rows, qa_top1=qa_hits / qa_total, qa_rows=qa_total)


def run_trials(mid, train, partners, validation, replay, guard, out, device, steps=STEPS):
    from time import monotonic

    model = ContrastiveHeads(config.head, config.loss).to(device)
    model.load_checkpoint_weights(mid)
    model.log_logit_scale.requires_grad_(False)
    baseline = assess(model, validation, guard, device)
    selected = dict(recipe="mid", step=0, **baseline)
    history = [selected]
    cfg = dict(width=config.head.width, depth=3, projection_dim=config.head.projection_dim,
               activation="gelu", layernorm=True, residual=False, hidden_size=config.head.input_dim)

    def save():
        partial = out / "best.partial"
        torch.save({**model.checkpoint_weights(), "cfg": cfg, "selection": selected}, partial)
        partial.replace(out / "best.pt")

    save()
    print("mid", baseline, flush=True)
    for recipe in ("contrastive", "temporal_swap", "temporal_swap_state_only"):
        started = monotonic()
        model.load_checkpoint_weights(mid)
        model.action_head.requires_grad_(recipe != "temporal_swap_state_only")
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE, weight_decay=0)
        rng = np.random.default_rng(SEED)
        for step in range(1, steps + 1):
            rows = rng.choice(train.num_rows, AGENT_BATCH, replace=False)
            qa_rows = rng.choice(replay.num_rows, QA_BATCH, replace=False)
            agent = embed_batch(model, train.take(rows).to_device(device))
            qa = embed_batch(model, replay.take(qa_rows).to_device(device))
            loss = 0.6 * contrastive_loss(agent) + 0.4 * contrastive_loss(qa)
            if recipe != "contrastive":
                loss += 0.25 * swap_loss(model, train, partners, rows, agent, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            if step % CHECK_EVERY == 0 or step == steps:
                row = dict(recipe=recipe, step=step, **assess(model, validation, guard, device),
                           elapsed_seconds=round(monotonic() - started, 2))
                print(row, flush=True)
                history.append(row)
                if row["qa_top1"] >= baseline["qa_top1"] - QA_TOLERANCE and row["agent_top1"] > selected["agent_top1"]:
                    selected = row
                    save()
    return {"selected": selected, "history": history, "dataset": DATASET, "revision": REVISION,
            "seed": SEED, "init": "mid", "qa_tolerance": QA_TOLERANCE, "validation_rows": validation.num_rows,
            "agent_rows": train.num_rows, "replay_rows": replay.num_rows, "learning_rate": LEARNING_RATE,
            "agent_batch": AGENT_BATCH, "qa_batch": QA_BATCH, "steps_per_recipe": steps}


@app.function(**RESOURCES, gpu="L4", timeout=1800, max_containers=1)
def sweep():
    volume.reload()
    mid_path = ROOT / config.runs_dir / "mid/best.pt"
    mid = torch.load(mid_path, map_location="cpu", weights_only=True)
    train, partners, train_tasks = terminal_pairs("train")
    dev, _, dev_tasks = terminal_pairs("val")
    if train_tasks & dev_tasks:
        raise ValueError("Task identity occurs on both sides of the terminal-agent split")
    rng = np.random.default_rng(SEED)
    validation = dev.take(rng.choice(dev.num_rows, size=4096, replace=False))
    store = ChunkStore(config.data_dir)
    chunks = [ChunkId(shard="shard_00", index=i) for i in (0, 7, 14, 21)]
    if config.held_out.all_chunks.intersection(chunks):
        raise ValueError("Sweep replay overlaps configured final Q&A holdout")
    replay = replay_sample(store, chunks[:-1], rng, 8192)
    guard = replay_sample(store, chunks[-1:], rng, 4096)
    name = f"post-terminal-{uuid4().hex[:10]}"
    out = ROOT / config.runs_dir / name
    out.mkdir()
    torch.set_float32_matmul_precision("high")
    report = run_trials(mid, train, partners, validation, replay, guard, out, torch.device("cuda"))
    report.update(run=name, mid_step=mid.get("step"), replay_chunks=[c.index for c in chunks[:-1]],
                  guard_chunk=chunks[-1].index)
    (out / "selection.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def run():
    prepare.remote()
    print(json.dumps(sweep.remote(), indent=2))


@app.function(**RESOURCES, gpu=["L4", "A10G", "L40S"], timeout=900)
@torch.no_grad()
def final_qa(run: str):
    """Strict positive > hardest valid negative, matching the workshop's final_eval.py."""
    volume.reload()
    folder = ROOT / config.runs_dir / run
    store = ChunkStore(config.data_dir)
    models = {}
    for name, path in {"mid": ROOT / config.runs_dir / "mid/best.pt", "selected": folder / "best.pt",
                       "clm": ROOT / "heads/qwen3_8b_posttrained_head.pt"}.items():
        model = ContrastiveHeads(config.head, config.loss).cuda().eval()
        model.load_checkpoint_weights(torch.load(path, weights_only=True, map_location="cpu"))
        models[name] = model
    hits = dict.fromkeys(models, 0)
    total = 0
    for chunk in config.held_out.hard_negatives:
        pairs = store.read(chunk, with_hard_negatives=True, mmap_mode="r")
        for start in range(0, pairs.num_rows, 512):
            rows = np.arange(start, min(start + 512, pairs.num_rows))
            batch = pairs.take(rows).to_device(torch.device("cuda"))
            valid = batch.negative_counts > 0
            total += valid.sum().item()
            for name, model in models.items():
                logits = true_vs_hard_negative_logits(embed_batch(model, batch))[valid]
                hits[name] += (logits[:, 0] > logits[:, 1:].amax(dim=1)).sum().item()
    result = dict(rows=total, hits=hits, accuracy={name: hit / total for name, hit in hits.items()})
    (folder / "final_qa.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print(result, flush=True)


@app.local_entrypoint()
def compare_general_qa(first: str, final: str):
    """Score both locked general checkpoints against the identical official Q&A split."""
    for run in dict.fromkeys((first, final)):
        final_qa.remote(run)
