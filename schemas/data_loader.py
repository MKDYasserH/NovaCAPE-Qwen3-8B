import os
from typing import Literal
from pydantic import BaseModel


class Source(BaseModel):
    kind: str
    repo: str
    revision: str | None = None
    extensions: list[str]
    group_by_chunk: bool = False


class Heads(BaseModel):
    repo: str
    files: list[str]


class LocalSettings(BaseModel):
    data_dir: str
    jobs_per_source: int
    agent_subsets: int
    pairs_per_subset: int


class AgentPairsSettings(BaseModel):
    app_name: str
    repo: str
    revision: str | None = None
    recordings_glob: str  # inside each subset folder
    vision_subsets: list[str]  # screenshot-based agents: Qwen3-8B only reads text
    pairs_per_subset: int
    pairs_per_trajectory: int
    min_actions: int  # skip recordings where the agent acts fewer times than this
    max_chars: int  # per message and per action
    validation_percent: int
    output_dir: str


class AgentEmbeddingSettings(BaseModel):
    app_name: str
    model_repo: str
    model_revision: str
    model_dir: str
    output_dir: str
    max_model_len: int
    max_tokens: int  # per state or action, kept under max_model_len
    state_char_budget: int  # cheap cut by characters before tokenizing a long state
    texts_per_call: int
    gpu_memory_utilization: float


class Config(BaseModel):
    run_mode: Literal["local", "remote"]
    app_name: str
    volume_name: str
    hf_secret_name: str
    data_dir: str
    log_every: int
    local: LocalSettings
    sources: list[Source]
    heads: Heads
    agent_pairs: AgentPairsSettings
    agent_embedding: AgentEmbeddingSettings


class RemoteFile(BaseModel):
    path: str
    size: int


class Job(BaseModel):
    source: Source
    files: list[RemoteFile]

    def destination(self, data_dir: str, file: RemoteFile) -> str:
        return os.path.join(data_dir, self.source.kind, file.path)
