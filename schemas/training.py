from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict


# --- Config (finetuning/config.toml) ---


class ChunkId(BaseModel):
    # Frozen makes it hashable, so chunks can be put in a set.
    model_config = ConfigDict(frozen=True)

    shard: str
    index: int


class LocalSettings(BaseModel):
    data_dir: str
    batch_size: int
    max_steps: int


class ModalSettings(BaseModel):
    gpu: str
    cpu: int
    memory_mb: int
    timeout_s: int
    max_retries: int
    retry_delay_s: float


class HeadDims(BaseModel):
    input_dim: int
    width: int
    projection_dim: int


class LossSettings(BaseModel):
    initial_temperature: float
    max_logit_scale: float


class OptimizerSettings(BaseModel):
    weight_decay: float
    grad_clip_norm: float


class HeldOutSettings(BaseModel):
    general: list[ChunkId]
    hard_negatives: list[ChunkId]
    eval_batch_size: int
    hard_negative_eval_rows: int

    @property
    def all_chunks(self) -> set[ChunkId]:
        return set(self.general) | set(self.hard_negatives)


class StageSettings(BaseModel):
    batch_size: int
    learning_rate: float
    max_epochs: int
    chunks_per_pool: int
    pool_repeats: int
    eval_every_steps: int
    full_eval_every_steps: int | None = None  # None: every eval includes the hard-negative eval
    patience: int
    learn_temperature: bool = True
    init_from_run: str | None = None  # warm start from <runs_dir>/<init_from_run>/best.pt
    max_steps: int | None = None


class AgentDataSettings(BaseModel):
    qa_replay_fraction: float
    states_dir: str
    actions_dir: str


class PostStageSettings(StageSettings):
    agent: AgentDataSettings


class TrainingConfig(BaseModel):
    run_mode: Literal["local", "remote"]
    app_name: str
    volume_name: str
    data_dir: str
    runs_dir: str
    seed: int
    log_every_steps: int
    min_improvement: float
    local: LocalSettings
    modal: ModalSettings
    head: HeadDims
    loss: LossSettings
    optimizer: OptimizerSettings
    held_out: HeldOutSettings
    pre: StageSettings
    mid: StageSettings
    post: PostStageSettings


class StageRun(BaseModel):
    name: str  # also the folder under runs_dir, and what a later stage's init_from_run points to
    settings: StageSettings
    data_dir: str


class BatchStreamSettings(BaseModel):
    chunks: list[ChunkId]  # already in the order to read them
    chunks_per_pool: int
    pool_repeats: int
    batch_size: int
    with_hard_negatives: bool
    seed: int
    prefetch_batches: int = 8


# --- Data ---


class PairTensors(BaseModel):
    """PairArrays moved to the training device."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: torch.Tensor
    actions: torch.Tensor
    doc_ids: torch.Tensor
    hard_negatives: torch.Tensor | None = None
    negative_counts: torch.Tensor | None = None


class PairArrays(BaseModel):
    """Rows of (state, action) pairs in memory. Row i of every array belongs to the same pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: np.ndarray  # [rows, dim] from *_q.npy: the question, or the agent's state
    actions: np.ndarray  # [rows, dim] from *_a.npy: the answer, or the agent's action
    doc_ids: np.ndarray  # [rows] int64; pairs from one document are never each other's negatives
    hard_negatives: np.ndarray | None = None  # [rows, k, dim] wrong actions that look right
    negative_counts: np.ndarray | None = None  # [rows] how many of the k slots hold a real negative

    @property
    def num_rows(self) -> int:
        return len(self.states)

    def take(self, rows: np.ndarray) -> "PairArrays":
        return PairArrays(
            states=self.states[rows],
            actions=self.actions[rows],
            doc_ids=self.doc_ids[rows],
            hard_negatives=_select_rows(self.hard_negatives, rows),
            negative_counts=_select_rows(self.negative_counts, rows),
        )

    @classmethod
    def concat(cls, parts: list["PairArrays"]) -> "PairArrays":
        return cls(
            states=np.concatenate([part.states for part in parts]),
            actions=np.concatenate([part.actions for part in parts]),
            doc_ids=np.concatenate([part.doc_ids for part in parts]),
            hard_negatives=_join_optional([part.hard_negatives for part in parts]),
            negative_counts=_join_optional([part.negative_counts for part in parts]),
        )

    def to_device(self, device: torch.device) -> PairTensors:
        # Copy float16 to the GPU first, then upcast there: half the bytes to transfer.
        return PairTensors(
            states=torch.from_numpy(self.states).to(device).float(),
            actions=torch.from_numpy(self.actions).to(device).float(),
            doc_ids=torch.from_numpy(self.doc_ids).to(device),
            hard_negatives=None if self.hard_negatives is None else torch.from_numpy(self.hard_negatives).to(device).float(),
            negative_counts=None if self.negative_counts is None else torch.from_numpy(self.negative_counts).to(device).long(),
        )


def _select_rows(array: np.ndarray | None, rows: np.ndarray) -> np.ndarray | None:
    return None if array is None else array[rows]


def _join_optional(arrays: list[np.ndarray | None]) -> np.ndarray | None:
    present = [array for array in arrays if array is not None]
    if not present:
        return None
    if len(present) != len(arrays):
        raise ValueError("can't mix rows with and without hard negatives in one batch")
    return np.concatenate(present)


class EmbeddedBatch(BaseModel):
    """A batch after both heads, L2-normalized, plus the current logit scale."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: torch.Tensor  # [rows, projection_dim]
    actions: torch.Tensor  # [rows, projection_dim]
    doc_ids: torch.Tensor
    scale: torch.Tensor  # 1 / temperature
    hard_negatives: torch.Tensor | None = None  # [rows, k, projection_dim]
    negative_counts: torch.Tensor | None = None


# --- Results ---


class RetrievalMetrics(BaseModel):
    loss: float
    top1: float  # share of states whose own action scores highest among the batch


class HardNegativeMetrics(BaseModel):
    loss: float
    top1: float  # share of states whose own action beats all of their hard negatives
    rows: int


class EvalReport(BaseModel):
    step: int
    epoch: int
    logit_scale: float
    minutes: float
    general: RetrievalMetrics
    hard_negatives: HardNegativeMetrics | None = None
    agent: RetrievalMetrics | None = None
    is_best: bool = False


class TrainState(BaseModel):
    """Counters saved in latest.pt so a restarted run continues where it stopped."""

    step: int = 0
    epoch: int = 0
    best_score: float = float("inf")
    evals_without_improvement: int = 0


class TrainingSummary(BaseModel):
    stage: str
    step: int
    epoch: int
    best_score: float
    minutes: float
