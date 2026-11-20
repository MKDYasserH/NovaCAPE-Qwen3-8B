"""Symmetric contrastive (CLIP-style) loss: each state should pick its own action, and each action its own state."""
import torch
import torch.nn.functional as F

from finetuning.heads import ContrastiveHeads
from schemas.training import EmbeddedBatch, PairTensors

TRUE_ACTION_COLUMN = 0  # in true_vs_hard_negative_logits


def embed_batch(model: ContrastiveHeads, batch: PairTensors) -> EmbeddedBatch:
    hard_negatives = None
    if batch.hard_negatives is not None:
        rows, slots, dim = batch.hard_negatives.shape
        # Hard negatives are wrong actions, so they go through the action head too.
        hard_negatives = model.embed_actions(batch.hard_negatives.view(-1, dim)).view(rows, slots, -1)
    return EmbeddedBatch(
        states=model.embed_states(batch.states),
        actions=model.embed_actions(batch.actions),
        doc_ids=batch.doc_ids,
        scale=model.logit_scale(),
        hard_negatives=hard_negatives,
        negative_counts=batch.negative_counts,
    )


def in_batch_logits(embedded: EmbeddedBatch) -> torch.Tensor:
    """[rows, rows]: row i scores state i against every action. Pair i's own action is column i."""
    logits = embedded.scale * embedded.states @ embedded.actions.t()
    same_doc = embedded.doc_ids[None] == embedded.doc_ids[:, None]
    same_doc.fill_diagonal_(False)  # keep each pair's own action
    # Another pair from the same document is probably a fine answer too, so don't count it as wrong.
    return logits.masked_fill(same_doc, float("-inf"))


def hard_negative_logits(embedded: EmbeddedBatch) -> torch.Tensor:
    """[rows, k]: each state against its own hard negatives. Empty slots are -inf."""
    logits = embedded.scale * torch.einsum("rd,rkd->rk", embedded.states, embedded.hard_negatives)
    slot = torch.arange(logits.shape[1], device=logits.device)
    return logits.masked_fill(slot[None] >= embedded.negative_counts[:, None], float("-inf"))


def true_vs_hard_negative_logits(embedded: EmbeddedBatch) -> torch.Tensor:
    """[rows, 1 + k]: the true action, then the hard negatives. No other pairs' actions."""
    true_logits = embedded.scale * (embedded.states * embedded.actions).sum(-1, keepdim=True)
    return torch.cat([true_logits, hard_negative_logits(embedded)], dim=1)


def contrastive_loss(embedded: EmbeddedBatch) -> torch.Tensor:
    logits = in_batch_logits(embedded)
    targets = torch.arange(len(logits), device=logits.device)
    # Read by column: for each action, which state is its own?
    action_to_state = F.cross_entropy(logits.t(), targets)
    # Read by row: for each state, which action is its own? Hard negatives are extra wrong actions for that row.
    if embedded.hard_negatives is not None:
        logits = torch.cat([logits, hard_negative_logits(embedded)], dim=1)
    state_to_action = F.cross_entropy(logits, targets)
    return (state_to_action + action_to_state) / 2


def count_top1_hits(embedded: EmbeddedBatch) -> int:
    """How many states score their own action highest among the batch."""
    logits = in_batch_logits(embedded)
    targets = torch.arange(len(logits), device=logits.device)
    return (logits.argmax(dim=1) == targets).sum().item()
