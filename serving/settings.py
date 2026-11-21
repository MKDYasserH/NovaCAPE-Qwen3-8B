"""The serving config, and what it shares with the rest of the project: the Qwen3-8B revision, the head sizes, the CLM commit."""
import tomllib
from pathlib import Path

from schemas.data_loader import Config as DataConfig
from schemas.evaluation import EvaluationConfig
from schemas.serving import ServingConfig
from schemas.training import TrainingConfig

PROJECT_DIR = Path(__file__).resolve().parent.parent  # the repo, or /root in a Modal container
CONFIG_NAMES = ["serving/config.toml", "data/config.toml", "finetuning/config.toml", "evaluation/config.toml"]


def read_toml(name: str) -> dict:
    with (PROJECT_DIR / name).open("rb") as file:
        return tomllib.load(file)


config = ServingConfig.model_validate(read_toml("serving/config.toml"))
encoder_model = DataConfig.model_validate(read_toml("data/config.toml")).agent_embedding  # the Qwen3-8B revision of training
training = TrainingConfig.model_validate(read_toml("finetuning/config.toml"))  # head sizes and the scale cap
clm = EvaluationConfig.model_validate(read_toml("evaluation/config.toml")).deepswe  # the CLM commit of the evaluations
