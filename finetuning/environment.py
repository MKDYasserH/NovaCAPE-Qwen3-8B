"""Config, Modal resources and device choice shared by the three training stages."""
import tomllib
from pathlib import Path

import modal

from schemas.training import StageSettings, TrainingConfig

PROJECT_DIR = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent / "config.toml"

with CONFIG_PATH.open("rb") as config_file:
    config = TrainingConfig.model_validate(tomllib.load(config_file))

IS_LOCAL = config.run_mode == "local"
data_dir = config.local.data_dir if IS_LOCAL else config.data_dir

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(str(PROJECT_DIR))
    .add_local_file(CONFIG_PATH, f"/root/finetuning/{CONFIG_PATH.name}")
    .add_local_python_source("schemas", "finetuning")
)
volume = modal.Volume.from_name(config.volume_name, create_if_missing=True, version=2)
app = modal.App(config.app_name)

# Decorator for each stage's entry function.
gpu_function = app.function(
    image=image,
    volumes={config.data_dir: volume},
    gpu=config.modal.gpu,
    cpu=config.modal.cpu,
    memory=config.modal.memory_mb,
    timeout=config.modal.timeout_s,
    retries=modal.Retries(max_retries=config.modal.max_retries, initial_delay=config.modal.retry_delay_s),
)


def reload_volume() -> None:
    if not IS_LOCAL:
        volume.reload()


def commit_volume() -> None:
    if not IS_LOCAL:
        volume.commit()


def stage_settings[S: StageSettings](settings: S) -> S:
    """In local mode, shrink the batch and stop after a few steps."""
    if not IS_LOCAL:
        return settings
    return settings.model_copy(update={"batch_size": config.local.batch_size, "max_steps": config.local.max_steps})
