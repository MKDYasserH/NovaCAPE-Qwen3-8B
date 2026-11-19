"""Stage 3: continue from mid's best.pt on agent steps, mixing in Q&A so the heads don't forget it.

Needs agent step embeddings in <data_dir>/agent/vec (the original repo makes them with pipeline/embed_agent.py).

    modal run -m finetuning.post_training_handler    (add --detach for a remote run)
"""
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from finetuning.chunks import hash_ids
from finetuning.environment import IS_LOCAL, app, config, data_dir, gpu_function, reload_volume, stage_settings
from finetuning.training_handler import Trainer
from schemas.training import AgentDataSettings, EvalReport, PairArrays, StageRun, TrainingConfig

VALIDATION_SPLIT = "val"  # value of the `split` column for held-out agent steps


class AgentSteps(BaseModel):
    train: PairArrays
    validation: PairArrays


def load_agent_steps(data_dir: str, settings: AgentDataSettings) -> AgentSteps:
    states_dir = Path(data_dir) / settings.states_dir
    actions_dir = Path(data_dir) / settings.actions_dir
    meta_files = sorted(states_dir.glob("*_meta.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"no agent steps under {states_dir}")
    prefixes = [path.name.removesuffix("_meta.parquet") for path in meta_files]
    meta = pa.concat_tables([pq.read_table(path, columns=["traj", "split", "config"]) for path in meta_files])
    # Steps from one trajectory are masked from each other, like pairs from one document.
    run_configs = meta.column("config").to_pylist()
    trajectories = meta.column("traj").to_pylist()
    trajectory_ids = [run_config + trajectory for run_config, trajectory in zip(run_configs, trajectories)]
    steps = PairArrays(
        states=np.concatenate([np.load(states_dir / f"{prefix}_q.npy") for prefix in prefixes]),
        actions=np.concatenate([np.load(actions_dir / f"{prefix}_a.npy") for prefix in prefixes]),
        doc_ids=hash_ids(trajectory_ids),
    )
    is_validation = np.array(meta.column("split").to_pylist()) == VALIDATION_SPLIT
    return AgentSteps(train=steps.take(np.flatnonzero(~is_validation)), validation=steps.take(np.flatnonzero(is_validation)))


class PostTrainer(Trainer):
    def __init__(self, run: StageRun, config: TrainingConfig):
        super().__init__(run, config)
        self.agent_settings = config.post.agent
        self.agent = load_agent_steps(run.data_dir, self.agent_settings)
        print(f"agent: {self.agent.train.num_rows} train steps, {self.agent.validation.num_rows} validation steps", flush=True)

    @contextmanager
    def epoch_batches(self) -> Iterator[Iterable[PairArrays]]:
        qa_rows = int(self.settings.batch_size * self.agent_settings.qa_replay_fraction)
        rng = np.random.default_rng(self.config.seed + self.state.epoch)
        with self.qa_batches(qa_rows) as qa_batches:
            yield (self._add_agent_steps(qa_batch, rng) for qa_batch in qa_batches)

    def _add_agent_steps(self, qa_batch: PairArrays, rng: np.random.Generator) -> PairArrays:
        """Fill the rest of the batch with agent steps, sampled with replacement."""
        agent_rows = self.settings.batch_size - qa_batch.num_rows
        rows = np.sort(rng.integers(0, self.agent.train.num_rows, agent_rows))
        return PairArrays.concat([qa_batch, self.agent.train.take(rows)])

    def evaluate(self, full_eval: bool) -> EvalReport:
        report = super().evaluate(full_eval)
        report.agent = self.evaluator.retrieval(self.model, self.agent.validation)
        return report

    def validation_score(self, report: EvalReport) -> float:
        return report.agent.loss


@gpu_function
def train_post() -> dict:
    reload_volume()
    run = StageRun(name="post", settings=stage_settings(config.post), data_dir=data_dir)
    return PostTrainer(run, config).train().model_dump()


@app.local_entrypoint()
def run_post():
    print(train_post.local() if IS_LOCAL else train_post.remote())
