"""Runs pre -> mid -> post for a few steps on tiny fake data, to check the whole pipeline on a laptop.

    uv run python -m finetuning.smoke_test
"""
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from finetuning.environment import config
from finetuning.mid_training_handler import MidTrainer
from finetuning.post_training_handler import PostTrainer
from finetuning.pre_training_handler import PreTrainer
from schemas.training import ChunkId, HeadDims, StageRun, StageSettings, TrainingConfig

DIMS = HeadDims(input_dim=32, width=64, projection_dim=16)
ROWS_PER_CHUNK = 96
DOCS_PER_CHUNK = 24
NEGATIVE_SLOTS = 10
AGENT_STEPS_PER_FILE = 120
rng = np.random.default_rng(0)


def random_vectors(*shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float16)


def write_chunk(data_dir: Path, chunk: ChunkId, with_hard_negatives: bool) -> None:
    prefix = f"{chunk.shard}/chunk_{chunk.index:04d}"
    (data_dir / "qa" / chunk.shard).mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "qa" / f"{prefix}_q.npy", random_vectors(ROWS_PER_CHUNK, DIMS.input_dim))
    np.save(data_dir / "qa" / f"{prefix}_a.npy", random_vectors(ROWS_PER_CHUNK, DIMS.input_dim))
    doc_ids = [f"{prefix}/doc{row % DOCS_PER_CHUNK}" for row in range(ROWS_PER_CHUNK)]
    pq.write_table(pa.table({"doc_id": doc_ids}), data_dir / "qa" / f"{prefix}_meta.parquet")
    if with_hard_negatives:
        (data_dir / "neg" / chunk.shard).mkdir(parents=True, exist_ok=True)
        np.save(data_dir / "neg" / f"{prefix}_n.npy", random_vectors(ROWS_PER_CHUNK, NEGATIVE_SLOTS, DIMS.input_dim))
        counts = rng.integers(0, NEGATIVE_SLOTS + 1, ROWS_PER_CHUNK).astype(np.uint8)
        np.save(data_dir / "neg" / f"{prefix}_ncnt.npy", counts)


def write_agent_steps(data_dir: Path) -> None:
    agent_dir = data_dir / config.post.agent.states_dir
    agent_dir.mkdir(parents=True, exist_ok=True)
    for name in ["subset_a", "subset_b"]:
        np.save(agent_dir / f"{name}_q.npy", random_vectors(AGENT_STEPS_PER_FILE, DIMS.input_dim))
        np.save(agent_dir / f"{name}_a.npy", random_vectors(AGENT_STEPS_PER_FILE, DIMS.input_dim))
        meta = pa.table({
            "traj": [f"traj{row % 12}" for row in range(AGENT_STEPS_PER_FILE)],
            "split": ["val" if row % 3 == 0 else "train" for row in range(AGENT_STEPS_PER_FILE)],
            "subset": [name] * AGENT_STEPS_PER_FILE,
        })
        pq.write_table(meta, agent_dir / f"{name}_meta.parquet")


def write_fake_data(data_dir: Path) -> None:
    for index in range(4):
        write_chunk(data_dir, ChunkId(shard="shard_00", index=index), with_hard_negatives=True)
        write_chunk(data_dir, ChunkId(shard="shard_01", index=index), with_hard_negatives=False)
    for chunk in config.held_out.hard_negatives:
        write_chunk(data_dir, chunk, with_hard_negatives=True)
    for chunk in config.held_out.general:
        write_chunk(data_dir, chunk, with_hard_negatives=False)
    write_agent_steps(data_dir)


def tiny(settings: StageSettings) -> StageSettings:
    return settings.model_copy(update={
        "batch_size": 32, "learning_rate": 1e-3, "chunks_per_pool": 2, "pool_repeats": 2,
        "eval_every_steps": 5, "max_epochs": 2, "max_steps": 12,
        "full_eval_every_steps": 10 if settings.full_eval_every_steps else None,
    })


def tiny_config() -> TrainingConfig:
    held_out = config.held_out.model_copy(update={"eval_batch_size": 32, "hard_negative_eval_rows": 50})
    return config.model_copy(update={"head": DIMS, "held_out": held_out, "log_every_steps": 5})


def main() -> None:
    small = tiny_config()
    with tempfile.TemporaryDirectory() as temp_dir:
        write_fake_data(Path(temp_dir))
        print(PreTrainer(StageRun(name="pre", settings=tiny(small.pre), data_dir=temp_dir), small).train())
        print(MidTrainer(StageRun(name="mid", settings=tiny(small.mid), data_dir=temp_dir), small).train())
        print(PostTrainer(StageRun(name="post", settings=tiny(small.post), data_dir=temp_dir), small).train())
        # A second run of a finished stage should resume from latest.pt and stop right away.
        print(PreTrainer(StageRun(name="pre", settings=tiny(small.pre), data_dir=temp_dir), small).train())


if __name__ == "__main__":
    main()
