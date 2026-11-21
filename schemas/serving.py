"""Config for running the decision model (serving/), and the question and answer of one decision."""
from typing import Literal

from pydantic import BaseModel, Field


class HeadSource(BaseModel):
    """The published heads, used wherever the model runs."""
    hf_repo: str
    hf_file: str
    hf_revision: str | None = None  # a commit hash pins the exact checkpoint


class LocalModelSettings(BaseModel):
    """serving/decision_model.py: Qwen3-8B through transformers, on this machine."""
    device: Literal["auto", "cuda", "mps", "cpu"]
    dtype: Literal["bfloat16", "float16", "float32"]
    batch_size: int  # texts per forward pass
    max_tokens: int  # longer texts keep their last tokens, as in training


class EncoderSettings(BaseModel):
    """The vLLM server that turns text into Qwen3-8B embeddings on Modal."""
    port: int
    gpu_memory_utilization: float
    max_model_len: int


class ModalSettings(BaseModel):
    """serving/server.py: the optional HTTP server on a Modal GPU."""
    app_name: str
    volume_name: str
    data_dir: str
    hf_secret_name: str | None = None  # a Modal secret holding HF_TOKEN; optional, everything used is public
    hf_cache_dir: str  # relative to data_dir, so downloads are reused
    gpu: str
    scaledown_window_s: int
    startup_timeout_s: int
    heads_port: int
    encoder: EncoderSettings


class ServingConfig(BaseModel):
    heads: HeadSource
    local: LocalModelSettings
    modal: ModalSettings


class Option(BaseModel):
    name: str
    description: str  # what the action head reads


class ChoiceQuestion(BaseModel):
    instructions: str
    options: list[Option] = Field(min_length=2)


class OptionProbability(BaseModel):
    name: str
    probability: float


class Decision(BaseModel):
    probabilities: list[OptionProbability]
    latency_ms: float  # time to answer, network included when the model is remote

    @property
    def chosen(self) -> str:
        return max(self.probabilities, key=lambda option: option.probability).name
