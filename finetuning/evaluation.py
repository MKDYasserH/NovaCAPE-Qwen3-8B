"""Scores the heads on chunks they never train on."""
import numpy as np
import torch
import torch.nn.functional as F

from finetuning.chunks import ChunkStore
from finetuning.heads import ContrastiveHeads
from finetuning.loss import TRUE_ACTION_COLUMN, contrastive_loss, count_top1_hits, embed_batch, true_vs_hard_negative_logits
from schemas.training import HardNegativeMetrics, HeldOutSettings, PairArrays, RetrievalMetrics

EVAL_SHUFFLE_SEED = 0  # fixed, so every checkpoint is scored on the same batches


class Evaluator:
    def __init__(self, store: ChunkStore, settings: HeldOutSettings, device: torch.device):
        self.settings = settings
        self.device = device
        # Small enough to keep in memory for the whole run.
        self.general_pairs = PairArrays.concat([store.read(chunk, with_hard_negatives=False) for chunk in settings.general])
        self.hard_negative_pairs = PairArrays.concat(
            [store.read(chunk, with_hard_negatives=True) for chunk in settings.hard_negatives]
        )

    def general(self, model: ContrastiveHeads) -> RetrievalMetrics:
        return self.retrieval(model, self.general_pairs)

    @torch.no_grad()
    def retrieval(self, model: ContrastiveHeads, pairs: PairArrays) -> RetrievalMetrics:
        """The training loss and top-1, in batches of eval_batch_size held-out pairs."""
        batch_size = self.settings.eval_batch_size
        if pairs.num_rows < batch_size:
            raise ValueError(f"{pairs.num_rows} held-out rows, fewer than eval_batch_size={batch_size}")
        order = np.random.default_rng(EVAL_SHUFFLE_SEED).permutation(pairs.num_rows)
        losses, hits = [], 0
        model.eval()
        for start in range(0, pairs.num_rows - batch_size + 1, batch_size):
            rows = np.sort(order[start : start + batch_size])
            embedded = embed_batch(model, pairs.take(rows).to_device(self.device))
            losses.append(contrastive_loss(embedded).item())
            hits += count_top1_hits(embedded)
        model.train()
        return RetrievalMetrics(loss=float(np.mean(losses)), top1=hits / (len(losses) * batch_size))

    @torch.no_grad()
    def hard_negatives(self, model: ContrastiveHeads) -> HardNegativeMetrics:
        """Each state chooses between its own action and its own hard negatives only."""
        pairs, step = self.hard_negative_pairs, self.settings.hard_negative_eval_rows
        losses, hits, scored_rows = [], 0, 0
        model.eval()
        for start in range(0, pairs.num_rows, step):
            rows = np.arange(start, min(start + step, pairs.num_rows))
            embedded = embed_batch(model, pairs.take(rows).to_device(self.device))
            # A row with no hard negatives would be a free win, so leave it out.
            logits = true_vs_hard_negative_logits(embedded)[embedded.negative_counts > 0]
            if len(logits) == 0:
                continue
            targets = torch.full((len(logits),), TRUE_ACTION_COLUMN, device=logits.device)
            losses.append(F.cross_entropy(logits, targets).item())
            hits += (logits.argmax(dim=1) == targets).sum().item()
            scored_rows += len(logits)
        model.train()
        return HardNegativeMetrics(loss=float(np.mean(losses)), top1=hits / scored_rows, rows=scored_rows)
