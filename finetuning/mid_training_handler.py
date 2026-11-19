"""Stage 2: continue from pre's best.pt, adding up to k hard negatives per pair.

Hard negatives exist only for shard_00, so this trains on its chunks minus the held-out ones.

    modal run -m finetuning.mid_training_handler    (add --detach for a remote run)
"""
from finetuning.environment import IS_LOCAL, app, config, data_dir, gpu_function, reload_volume, stage_settings
from finetuning.training_handler import Trainer
from schemas.training import EvalReport, StageRun


class MidTrainer(Trainer):
    uses_hard_negatives = True

    def validation_score(self, report: EvalReport) -> float:
        if report.hard_negatives is None:
            raise ValueError("mid is scored on hard negatives: leave full_eval_every_steps unset in [mid]")
        return report.hard_negatives.loss


@gpu_function
def train_mid() -> dict:
    reload_volume()
    run = StageRun(name="mid", settings=stage_settings(config.mid), data_dir=data_dir)
    return MidTrainer(run, config).train().model_dump()


@app.local_entrypoint()
def run_mid():
    print(train_mid.local() if IS_LOCAL else train_mid.remote())
