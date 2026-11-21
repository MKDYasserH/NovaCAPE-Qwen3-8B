"""Turn Agent Data Protocol recordings into (state, action) text pairs for post-training.

state = the conversation up to an agent step, action = what the agent did at that step.
The authors' selection isn't public; this follows the original repo's reconstruction
(see [agent_pairs] in config.toml). Output: <data_dir>/agent/pairs/<subset>.parquet

    modal run data/load_agent_data.py    (run_mode = "local" builds a small sample on this machine)
"""
import os
import random
import tomllib
import zlib
from collections.abc import Iterator
from pathlib import Path

import modal
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from pydantic import ValidationError

from schemas.agent_data import VALIDATION_SPLIT, AgentPair, ChatMessage, Recording, Split, SubsetReport
from schemas.data_loader import Config

PROJECT_DIR = Path(__file__).parent.parent
# Modal mounts this script at /root, so the config sits next to it in the container too.
CONFIG_PATH = Path(__file__).parent / "config.toml"

with CONFIG_PATH.open("rb") as config_file:
    config = Config.model_validate(tomllib.load(config_file))

IS_LOCAL = config.run_mode == "local"
data_dir = config.local.data_dir if IS_LOCAL else config.data_dir
settings = config.agent_pairs
# In local mode, keep only a few hundred pairs per subset.
pairs_quota = config.local.pairs_per_subset if IS_LOCAL else settings.pairs_per_subset

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(str(PROJECT_DIR))
    .add_local_file(CONFIG_PATH, f"/root/{CONFIG_PATH.name}")
    .add_local_python_source("schemas")
)
volume = modal.Volume.from_name(config.volume_name, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name(config.hf_secret_name)  # sets HF_TOKEN, which huggingface_hub reads itself
app = modal.App(settings.app_name)

revision = f"@{settings.revision}" if settings.revision else ""
REPO_ROOT = f"datasets/{settings.repo}{revision}"  # HfFileSystem path of the dataset


def list_subsets() -> list[str]:
    """ADP calls each source dataset a "config"; we say subset, to keep config for config.toml."""
    folders = [entry["name"] for entry in HfFileSystem().ls(REPO_ROOT, detail=True) if entry["type"] == "directory"]
    subsets = sorted(Path(folder).name for folder in folders)
    return [subset for subset in subsets if subset not in settings.vision_subsets]


def output_path(subset: str) -> Path:
    return Path(data_dir) / settings.output_dir / f"{subset}.parquet"


def assign_split(trajectory_id: str) -> Split:
    """Hashing the id puts all of a trajectory's steps in one split, the same one on every run."""
    is_validation = zlib.crc32(trajectory_id.encode()) % 100 < settings.validation_percent
    return VALIDATION_SPLIT if is_validation else "train"


class SubsetBuilder:
    """Streams one subset's recordings and keeps up to pairs_quota pairs from them."""

    def __init__(self, subset: str):
        self.subset = subset
        self.rng = random.Random(zlib.crc32(subset.encode()))  # same pairs on every run
        self.report = SubsetReport(subset=subset, pairs=0)

    def build(self) -> list[AgentPair]:
        pairs: list[AgentPair] = []
        for recording in self._recordings():
            pairs.extend(self._pairs_from(recording))
            if len(pairs) >= pairs_quota:
                break
        pairs = pairs[:pairs_quota]
        self.report.pairs = len(pairs)
        return pairs

    def _recordings(self) -> Iterator[Recording]:
        """One recording per jsonl line. Files are streamed so we stop reading at the quota."""
        fs = HfFileSystem()
        for path in sorted(fs.glob(f"{REPO_ROOT}/{self.subset}/{settings.recordings_glob}")):
            # block_size=0 streams the file in one request instead of many range requests.
            with fs.open(path, "r", block_size=0) as lines:
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        recording = Recording.model_validate_json(line)
                    except ValidationError:
                        self.report.recordings_skipped += 1
                        continue
                    self.report.recordings_read += 1
                    yield recording

    def _pairs_from(self, recording: Recording) -> list[AgentPair]:
        """Replay the recording as a chat, and pair every agent action with the chat before it."""
        actions_taken = sum(1 for step in recording.steps if step.source == "agent" and step.action_text())
        if actions_taken < settings.min_actions:
            return []
        trajectory_id = recording.trajectory_id or recording.session_id or str(self.report.recordings_read)
        split = assign_split(trajectory_id)
        history: list[ChatMessage] = []
        pairs: list[AgentPair] = []
        for index, step in enumerate(recording.steps):
            if step.source in ("system", "user") and step.message:
                history.append(ChatMessage(role=step.source, content=self._clip(step.message)))
            if step.source != "agent":
                continue
            if action := self._clip(step.action_text()):
                if history:
                    pairs.append(AgentPair(
                        subset=self.subset, traj=trajectory_id, step=index, split=split, state=list(history), action=action
                    ))
                history.append(ChatMessage(role="assistant", content=action))
            # The environment's answer comes back to the agent as a user turn.
            if observation := self._clip(step.observation_text()):
                history.append(ChatMessage(role="user", content=observation))
        # Steps of one trajectory are near duplicates: keep a few, spread over the whole run.
        self.rng.shuffle(pairs)
        return pairs[: settings.pairs_per_trajectory]

    @staticmethod
    def _clip(text: str) -> str:
        return text[: settings.max_chars]


def write_pairs(pairs: list[AgentPair], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    pq.write_table(pa.Table.from_pylist([pair.to_row() for pair in pairs]), partial, compression="zstd")
    os.replace(partial, destination)


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], cpu=2, memory=8192, timeout=7200, retries=modal.Retries(max_retries=3, initial_delay=30.0), max_containers=24)
def build_subset(subset: str) -> SubsetReport:
    destination = output_path(subset)
    if destination.exists():
        return SubsetReport(subset=subset, pairs=pq.read_metadata(destination).num_rows, already_built=True)
    builder = SubsetBuilder(subset)
    pairs = builder.build()
    if pairs:
        write_pairs(pairs, destination)
        if not IS_LOCAL:
            volume.commit()
    print(builder.report, flush=True)
    return builder.report


def build_locally(subsets: list[str]) -> Iterator[SubsetReport | Exception]:
    # Mirrors build_subset.map(return_exceptions=True): a failed subset is reported, not raised.
    for subset in subsets:
        try:
            yield build_subset.local(subset)
        except Exception as error:
            yield error


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], timeout=14000)
def orchestrate():
    subsets = list_subsets()
    if IS_LOCAL:
        subsets = subsets[: config.local.agent_subsets]
    print(f"{len(subsets)} subsets", flush=True)
    results = build_locally(subsets) if IS_LOCAL else build_subset.map(subsets, order_outputs=False, return_exceptions=True)
    reports = []
    for result in results:
        if isinstance(result, Exception):
            print("subset failed", repr(result)[:200], flush=True)
            continue
        reports.append(result)
    print(f"DONE {sum(r.pairs for r in reports)} pairs from {sum(1 for r in reports if r.pairs)} subsets", flush=True)


@app.local_entrypoint()
def main():
    if IS_LOCAL:
        orchestrate.local()
    else:
        orchestrate.remote()
