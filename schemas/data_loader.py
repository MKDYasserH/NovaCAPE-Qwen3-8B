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


class Config(BaseModel):
    run_mode: Literal["local", "remote"]
    app_name: str
    volume_name: str
    data_dir: str
    log_every: int
    local: LocalSettings
    sources: list[Source]
    heads: Heads


class RemoteFile(BaseModel):
    path: str
    size: int


class Job(BaseModel):
    source: Source
    files: list[RemoteFile]

    def destination(self, data_dir: str, file: RemoteFile) -> str:
        return os.path.join(data_dir, self.source.kind, file.path)
