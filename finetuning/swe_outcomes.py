"""Coding-specific post-training from same-issue passing and failing SWE-bench rollouts.

Run via evaluation/deepswe.py::train_swe_transfer in the pinned CLM image.
"""
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.nn import functional as F

DATASET = "tarsur385/swebench-verified-top15-embeddings-8k"
REVISION = "d180d04450d87a0a1c4c3c185f64dcb45359af9c"
SEED = 1234
WINDOW = 12


def make_rollouts(samples, excluded):
    steps = defaultdict(list)
    for i, sample in enumerate(samples):
        if sample["task_id"] in excluded:
            raise ValueError("Source task matches a prohibited DeepSWE task ID")
        steps[sample["trajectory_id"]].append((sample["step_idx"], i))
    groups = {split: defaultdict(list) for split in ("train", "dev", "test")}
    rows, masks, labels = [], [], []
    for trajectory in steps.values():
        trajectory.sort()
        ids = [row for _, row in trajectory]
        keys = {(samples[i]["task_id"], samples[i]["config"], samples[i]["reward"]) for i in ids}
        if len(keys) != 1 or len({step for step, _ in trajectory}) != len(ids):
            raise ValueError("Inconsistent rollout metadata")
        task, source_split, reward = keys.pop()
        if reward not in (0, 1) or source_split not in ("swebench_verified/train", "swebench_verified/val"):
            raise ValueError(f"Unexpected split or label: {source_split}, {reward}")
        bucket = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 5
        split = "test" if source_split.endswith("/val") else ("dev" if bucket == 0 else "train")
        groups[split][task].append(len(rows))
        tail = ids[-WINDOW:]
        rows.append(tail + [tail[-1]] * (WINDOW - len(tail)))
        masks.append([1] * len(tail) + [0] * (WINDOW - len(tail)))
        labels.append(int(reward))
    train, dev, test = (set(g) for g in groups.values())
    if not (train and dev and test) or train & dev or train & test or dev & test:
        raise ValueError("Expected three nonempty, task-disjoint splits")
    return np.array(rows), torch.tensor(masks, dtype=torch.float32), np.array(labels), groups


def train(folder, init, source, exclusions):
    sys.path.insert(0, "/clm/evaluation")
    from bon_eval import best_of_n, load_heads

    samples = json.loads((source / "metadata.json").read_text())["samples"]
    rows, masks, labels, groups = make_rollouts(samples, exclusions)
    device = torch.device("cuda")
    masks = masks.to(device)
    matrices = []
    for name in ("state", "action"):
        matrix = torch.load(source / f"{name}_embeddings.pt", mmap=True, weights_only=True)
        matrices.append(matrix[rows].to(device))
    sh, ah, cfg = load_heads(str(init), device)
    initial = torch.load(init, weights_only=True, map_location="cpu")
    params = list(sh.parameters()) + list(ah.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.00004, weight_decay=0.0)
    rng = np.random.default_rng(SEED)

    def scores(ids):
        idx = torch.as_tensor(ids, device=device)
        branch = []
        for head, matrix in zip((sh, ah), matrices):
            branch.append(F.normalize(head(matrix[idx].flatten(0, 1).float()), dim=-1))
        cosine = (branch[0] * branch[1]).sum(-1).reshape(len(idx), WINDOW)
        return (cosine * masks[idx]).sum(1) / masks[idx].sum(1)

    @torch.no_grad()
    def evaluate(split):
        tasks = groups[split]
        ids = [i for task in tasks.values() for i in task]
        predicted = torch.cat([scores(ids[j:j+64]) for j in range(0, len(ids), 64)]).tolist()
        by_id = dict(zip(ids, predicted))
        metrics = [best_of_n([(by_id[i], int(labels[i])) for i in task], min(4, len(task))) for task in tasks.values()]
        return {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}

    pairs = []
    for ids in groups["train"].values():
        pos = [i for i in ids if labels[i] == 1]
        neg = [i for i in ids if labels[i] == 0]
        if pos and neg:
            pairs.append((pos, neg))
    if not pairs:
        raise ValueError("No mixed-outcome training tasks")
    history, best, best_step = [], -1, 0
    for step in range(301):
        if step in (0, 50, 150, 300):
            dev = evaluate("dev")
            history.append(dict(step=step, **dev))
            print("SWE development", history[-1], flush=True)
            if dev["selected"] > best:
                best, best_step = dev["selected"], step
                torch.save(dict(state_head=sh.state_dict(), action_head=ah.state_dict(), cfg=cfg,
                                logit_scale=initial["logit_scale"], step=step), folder / "best.pt")
        if step == 300:
            break
        chosen = rng.integers(len(pairs), size=24)
        pos = [rng.choice(pairs[t][0]) for t in chosen]
        neg = [rng.choice(pairs[t][1]) for t in chosen]
        passed, failed = scores(pos + neg).chunk(2)
        loss = F.softplus(-16 * (passed - failed)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()

    # Lock the external-development winner before either test-set evaluation.
    winner = torch.load(folder / "best.pt", weights_only=True, map_location="cpu")
    sh.load_state_dict(winner["state_head"])
    ah.load_state_dict(winner["action_head"])
    selected_test = evaluate("test")
    sh.load_state_dict(initial["state_head"])
    ah.load_state_dict(initial["action_head"])
    result = dict(dataset=DATASET, revision=REVISION, init=str(init), seed=SEED, best_step=best_step,
                  mixed_train=len(pairs), history=history, test=selected_test, init_test=evaluate("test"),
                  task_counts={k: len(v) for k, v in groups.items()}, prohibited_task_ids=len(exclusions))
    (folder / "transfer.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == "__main__":
    sys.path.insert(0, "/clm/preprocessing")
    import hf_embeddings

    root, init, out, metadata = map(Path, sys.argv[1:])
    exclusions = {s["task_id"] for s in json.loads(metadata.read_text())["samples"]}
    cache = root / "post_inputs" / REVISION
    local = snapshot_download(DATASET, repo_type="dataset", revision=REVISION, local_dir=cache,
                              allow_patterns=["data/*.parquet"])
    with tempfile.TemporaryDirectory() as temp:
        source = Path(hf_embeddings.download(local, temp))
        train(out, init, source, exclusions)
