"""Stage 1: train both heads from scratch on every Q&A chunk except the held-out ones.

    modal run -m finetuning.pre_training_handler    (add --detach for a remote run)
"""
from finetuning.environment import IS_LOCAL, app, config, data_dir, gpu_function, reload_volume, stage_settings
from finetuning.training_handler import Trainer
from schemas.training import EvalReport, StageRun


class PreTrainer(Trainer):
    def validation_score(self, report: EvalReport) -> float:
        return report.general.loss


@gpu_function
def train_pre() -> dict:
    reload_volume()
    run = StageRun(name="pre", settings=stage_settings(config.pre), data_dir=data_dir)
    return PreTrainer(run, config).train().model_dump()


@app.local_entrypoint()
def run_pre():
    print(train_pre.local() if IS_LOCAL else train_pre.remote())
