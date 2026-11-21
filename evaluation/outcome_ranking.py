"""Fit small per-coordinate adjustments to two verifier heads using same-task rollout outcomes."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def mixed_rollouts(samples, allowed, window):
    """Keep chronological tails; discard tasks that offer no pass/fail preference."""
    trajectories = defaultdict(list)
    for row, sample in enumerate(samples):
        if sample["task_id"] in allowed:
            group = (sample["task_id"], sample.get("config") or "default")
            trajectories[group, sample["trajectory_id"]].append((sample["step_idx"], row))
    groups = defaultdict(list)
    for (group, _), steps in trajectories.items():
        steps.sort()
        labels = {samples[row]["reward"] for _, row in steps}
        if len(labels) != 1 or None in labels:
            raise ValueError("Missing or inconsistent outcome within a rollout")
        groups[group].append(([r for _, r in steps[-window:]], labels.pop() > 0.5))
    return [rollouts for rollouts in groups.values() if len({passed for _, passed in rollouts}) == 2]


class CoordinateAdjustment(nn.Module):
    """Identity at initialization. Both transformations fold into existing output layers."""
    def __init__(self, dim):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(2, dim))
        self.bias = nn.Parameter(torch.zeros(2, dim))

    def forward(self, projected, rms):
        changed = projected * (1 + self.gain[:, None, :]) + rms * self.bias[:, None, :]
        states, actions = F.normalize(changed, dim=-1)
        return (states * actions).sum(-1)

    @torch.no_grad()
    def fold_into(self, checkpoint, rms):
        for i, name in enumerate(("state_head", "action_head")):
            head = checkpoint[name]
            gain = 1 + self.gain[i].cpu()
            head["out.weight"] *= gain[:, None]
            head["out.bias"] *= gain
            head["out.bias"] += self.bias[i].cpu() * rms[i].cpu().item()


def fit(folder, embeddings, heldout_file, window):
    sys.path.insert(0, "/clm/evaluation")
    from bon_eval import load_heads

    split = json.loads((folder / "task_split.json").read_text())
    train, dev = set(split["train_tasks"]), set(split["val_tasks"])
    held_out = set(json.loads(heldout_file.read_text()))
    if train & dev or (train | dev) & held_out:
        raise ValueError("Overlapping DeepSWE splits")
    samples = json.loads((embeddings / "metadata.json").read_text())["samples"]
    train_groups = mixed_rollouts(samples, train, window)
    dev_groups = mixed_rollouts(samples, dev, window)
    init = folder / "best_head.pt"
    checkpoint = torch.load(init, weights_only=True, map_location="cpu")
    history = []
    best_step = 0
    if train_groups and dev_groups:
        # Cache raw head outputs once. Only the terminal windows of train/dev mixed tasks enter the fit.
        groups = train_groups + dev_groups
        rows, owners, group_indices = [], [], []
        n_rollouts = 0
        for group in groups:
            positives, negatives = [], []
            for tail, passed in group:
                rows.extend(tail)
                owners.extend([n_rollouts] * len(tail))
                (positives if passed else negatives).append(n_rollouts)
                n_rollouts += 1
            group_indices.append((positives, negatives))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sh, ah, _ = load_heads(str(init), device)
        with torch.no_grad():
            projected = []
            for name, head in (("state", sh), ("action", ah)):
                matrix = torch.load(embeddings / f"{name}_embeddings.pt", weights_only=True, mmap=True)
                projected.append(head(matrix[rows].float().to(device)))
            projected = torch.stack(projected)
            train_end = sum(len(tail) for group in train_groups for tail, _ in group)
            rms = projected[:, :train_end].square().mean((1, 2), keepdim=True).sqrt()
        adjust = CoordinateAdjustment(projected.size(-1)).to(device)
        optimizer = torch.optim.Adam(adjust.parameters(), lr=0.025)
        owners = torch.tensor(owners, device=device)
        counts = torch.bincount(owners)
        best_key = None
        for step in range(61):
            sims = adjust(projected, rms)
            scores = sims.new_zeros(n_rollouts).index_add(0, owners, sims) / counts
            losses, wins = [], []
            for pos, neg in group_indices:
                positive, negative = scores[pos], scores[neg]
                losses.append(F.softplus(-16 * (positive[:, None] - negative)).mean())
                with torch.no_grad():
                    top = torch.maximum(positive.max(), negative.max())
                    wins.append((positive == top).sum() / ((positive == top).sum() + (negative == top).sum()))
            if step in (0, 10, 30, 60):
                val_loss = torch.stack(losses[len(train_groups):]).mean().item()
                val_pick = torch.stack(wins[len(train_groups):]).mean().item()
                history.append(dict(step=step, val_pick=val_pick, val_pair_loss=val_loss))
                key = (val_pick, -val_loss)
                if best_key is None or key > best_key:
                    best_key, best_step = key, step
                    best_weights = {k: v.detach().clone() for k, v in adjust.state_dict().items()}
            if step < 60:
                loss = torch.stack(losses[:len(train_groups)]).mean()
                loss = loss + 0.1 * (adjust.gain.square().mean() + adjust.bias.square().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        adjust.load_state_dict(best_weights)
        adjust.fold_into(checkpoint, rms)
    report = dict(step=best_step, mixed_train=len(train_groups), mixed_validation=len(dev_groups), history=history)
    checkpoint["outcome_ranking"] = report
    torch.save(checkpoint, folder / "ranked_head.pt")
    (folder / "outcome_selection.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    fit(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
