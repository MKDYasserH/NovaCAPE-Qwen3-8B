"""Config and results for evaluation/deepswe.py."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue


class LocalSettings(BaseModel):
    data_dir: str


class DeepSWESettings(BaseModel):
    clm_repo_url: str
    clm_commit: str
    heads_repo: str
    eval_dataset: str
    authors_init_repo: str
    authors_init_file: str
    output_dir: str
    min_reward: float
    batch_size: int
    seed: int
    max_epochs: int
    patience: int
    candidates: int
    window: int
    expected_training_pairs: int
    expected_training_tasks: int
    printed_chars: int


class EvaluationConfig(BaseModel):
    run_mode: Literal["local", "remote"]
    app_name: str
    volume_name: str
    hf_secret_name: str
    data_dir: str
    runs_dir: str
    local: LocalSettings
    deepswe: DeepSWESettings


class DeepSWESample(BaseModel):
    """One step of a DeepSWE rollout. Other fields are kept, since finetune.py reads them back."""
    model_config = ConfigDict(extra="allow")

    task_id: str
    reward: float | None = None  # 1.0 when the rollout's final patch passed the task's tests


class DeepSWEResult(BaseModel):
    init: str
    seed: int
    training_pairs: int
    training_tasks: int
    best_of_n: list[str]  # bon_eval.py's score lines: success rate and tasks solved


class HeadSummary(BaseModel):
    """Everything in a verifier checkpoint except the weights."""
    other_keys: list[str]
    cfg: JsonValue = None
    log_logit_scale: float
    logit_scale: float
    epoch: JsonValue = None
    metrics: JsonValue = None
