"""DeepSWE: does the head help pick a coding agent's best attempt?

run_deepswe fine-tunes a verifier from a CLM head on passing DeepSWE steps, using the CLM repo's own
finetune.py, then bon_eval.py scores 4 rollouts per held-out task and keeps the best. The authors' head
gets 31/38 tasks; random picking gets about 28.
inspect_deepswe prints what a verifier checkpoint stores (settings, metrics) and the files next to it.

    modal run --detach evaluation/deepswe.py::run_deepswe --init post       # our runs/post/best.pt
    modal run --detach evaluation/deepswe.py::run_deepswe --init authors    # control: should come near 31/38
    modal run evaluation/deepswe.py::inspect_deepswe                        # the authors' verifier
    modal run evaluation/deepswe.py::inspect_deepswe --head post            # ours (run_mode = "remote")
"""
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import modal
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import TypeAdapter

from schemas.evaluation import DeepSWEResult, DeepSWESample, EvaluationConfig, HeadSummary

# Modal mounts this script at /root, so the config sits next to it in the container too.
CONFIG_PATH = Path(__file__).parent / "config.toml"

with CONFIG_PATH.open("rb") as config_file:
    config = EvaluationConfig.model_validate(tomllib.load(config_file))

IS_LOCAL = config.run_mode == "local"
settings = config.deepswe
CLM_DIR = Path("/clm")
AUTHORS = "authors"  # --init / --head value for the authors' checkpoints instead of one of our runs
WEIGHT_KEYS = {"state_head", "action_head"}
FILES_NEXT_TO_HEAD = ["config.json", "verification.json", "split_summary.json", "README.md", "result.json", "finetune_summary.json"]

# The CLM scripts were run with these exact versions, so this image doesn't reuse the project's uv environment.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1", "numpy==2.1.3", "pyarrow==18.1.0", "huggingface_hub>=0.34,<1.0", "requests==2.32.3",
        "transformers==4.57.6", "pydantic>=2.10",
    )
    .run_commands(f"git clone {settings.clm_repo_url} {CLM_DIR} && git -C {CLM_DIR} checkout {settings.clm_commit}")
    .add_local_file(CONFIG_PATH, f"/root/{CONFIG_PATH.name}")
    .add_local_python_source("schemas")
)
volume = modal.Volume.from_name(config.volume_name, create_if_missing=True, version=2)
app = modal.App(config.app_name)

samples_adapter = TypeAdapter(list[DeepSWESample])


def deepswe_dir(data_dir: str) -> Path:
    return Path(data_dir) / settings.output_dir


def download_authors_head(data_dir: str) -> Path:
    """The authors' verifier folder: best_head.pt, heldout_tasks.json and its settings."""
    return Path(snapshot_download(settings.heads_repo, local_dir=deepswe_dir(data_dir) / "authors_head"))


def run_clm_script(args: list[str]) -> list[str]:
    """Run a script from the CLM repo, streaming its output to the logs, and return its lines."""
    process = subprocess.Popen([sys.executable, *args], cwd=CLM_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip())
    if process.wait() != 0:
        raise RuntimeError(f"{args[0]} exited with code {process.returncode}")
    return lines


class DeepSWEJob:
    """One fine-tune and evaluation, written to <data_dir>/deepswe/<init>/seed_<seed>."""

    def __init__(self, init: str, seed: int):
        self.init = init
        self.seed = seed
        self.out_dir = deepswe_dir(config.data_dir) / init / f"seed_{seed}"
        self.heads_dir = download_authors_head(config.data_dir)
        self.held_out_file = self.heads_dir / "heldout_tasks.json"

    def run(self) -> DeepSWEResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        init_checkpoint = self._init_checkpoint()
        with tempfile.TemporaryDirectory() as training_dir:
            samples = self._write_training_set(Path(training_dir))
            run_clm_script([
                "train/finetune.py", "--task", "clm", "--init-ckpt", str(init_checkpoint), "--out-dir", str(self.out_dir),
                "--emb-dir", training_dir, "--holdout-tasks", str(self.held_out_file), "--batch", str(settings.batch_size),
                "--seed", str(self.seed), "--epochs", str(settings.max_epochs), "--patience", str(settings.patience),
            ])
        volume.commit()
        bon_lines = run_clm_script([
            "evaluation/bon_eval.py", "--hf-dataset", settings.eval_dataset, "--checkpoint", str(self.out_dir / "best_head.pt"),
            "--tasks-file", str(self.held_out_file), "--n", str(settings.candidates), "--window", str(settings.window),
            "--cache-dir", str(deepswe_dir(config.data_dir) / "cache"), "--output", str(self.out_dir / "bon.json"),
        ])
        result = DeepSWEResult(
            init=self.init, seed=self.seed, training_pairs=len(samples), training_tasks=len({s.task_id for s in samples}),
            best_of_n=[line for line in bon_lines if line.startswith("[bon]") and "%" in line],
        )
        (self.out_dir / "result.json").write_text(result.model_dump_json(indent=1))
        volume.commit()
        return result

    def _init_checkpoint(self) -> Path:
        if self.init == AUTHORS:
            return Path(hf_hub_download(settings.authors_init_repo, settings.authors_init_file))
        path = Path(config.data_dir) / config.runs_dir / self.init / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path}: train the {self.init} stage first")
        return path

    def _write_training_set(self, training_dir: Path) -> list[DeepSWESample]:
        """The authors' head records 22,576 pairs from 66 tasks: the passing steps of the training tasks,
        taken from the evaluation rollouts. The README's default dataset (405,919 steps) doesn't reproduce it."""
        sys.path.insert(0, str(CLM_DIR / "preprocessing"))
        import hf_embeddings  # from the CLM repo: parquet dataset -> state/action .pt files + metadata.json

        eval_dir = Path(hf_embeddings.download(settings.eval_dataset, str(deepswe_dir(config.data_dir) / "eval_embeddings")))
        volume.commit()
        samples = samples_adapter.validate_python(json.loads((eval_dir / "metadata.json").read_text())["samples"])
        held_out = set(json.loads(self.held_out_file.read_text()))
        rows = [
            row for row, sample in enumerate(samples)
            if sample.task_id not in held_out and (sample.reward or 0) > settings.min_reward
        ]
        kept = [samples[row] for row in rows]
        print(
            f"training pairs {len(kept)}, tasks {len({s.task_id for s in kept})} "
            f"(authors' head: {settings.expected_training_pairs}, {settings.expected_training_tasks})", flush=True
        )
        rows_tensor = torch.tensor(rows)
        for name in ("state_embeddings.pt", "action_embeddings.pt"):
            torch.save(torch.load(eval_dir / name)[rows_tensor], training_dir / name)
        metadata = {"num_samples": len(kept), "hidden_size": 4096, "samples": [sample.model_dump() for sample in kept]}
        (training_dir / "metadata.json").write_text(json.dumps(metadata))
        return kept


@app.function(image=image, volumes={config.data_dir: volume}, gpu="A10G", cpu=8, memory=64000, timeout=7200)
def train_and_evaluate(init: str, seed: int) -> DeepSWEResult:
    volume.reload()
    return DeepSWEJob(init, seed).run()


def summarize_head(path: Path) -> HeadSummary:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # no pickled code from a download
    log_scale = float(torch.as_tensor(checkpoint["logit_scale"]))
    return HeadSummary(
        other_keys=sorted(key for key in checkpoint if key not in WEIGHT_KEYS),
        cfg=json.loads(json.dumps(checkpoint.get("cfg"), default=str)),
        log_logit_scale=log_scale,
        logit_scale=float(torch.tensor(log_scale).exp()),
        epoch=checkpoint.get("epoch"),
        metrics=json.loads(json.dumps(checkpoint.get("metrics"), default=str)),
    )


@app.function(image=image, volumes={config.data_dir: volume}, timeout=600)
def inspect_head(head: str, seed: int) -> None:
    data_dir = config.local.data_dir if IS_LOCAL else config.data_dir
    if not IS_LOCAL:
        volume.reload()
    folder = download_authors_head(data_dir) if head == AUTHORS else deepswe_dir(data_dir) / head / f"seed_{seed}"
    print(summarize_head(folder / "best_head.pt").model_dump_json(indent=1))
    for name in FILES_NEXT_TO_HEAD:
        if (folder / name).exists():
            print(f"== {name}\n{(folder / name).read_text()[: settings.printed_chars]}")


@app.local_entrypoint()
def run_deepswe(init: str = "post", seed: int = settings.seed):
    print(train_and_evaluate.remote(init, seed))


@app.local_entrypoint()
def inspect_deepswe(head: str = AUTHORS, seed: int = settings.seed):
    if IS_LOCAL:
        inspect_head.local(head, seed)
    else:
        inspect_head.remote(head, seed)
