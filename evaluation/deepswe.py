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
from uuid import uuid4

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
    .add_local_file(Path(__file__).with_name("outcome_ranking.py"), "/root/outcome_ranking.py")
    .add_local_file(CONFIG_PATH.parent.parent / "finetuning/swe_outcomes.py", "/root/finetuning/swe_outcomes.py")
    .add_local_python_source("schemas")
)
volume = modal.Volume.from_name(config.volume_name, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name(config.hf_secret_name)  # sets HF_TOKEN, which huggingface_hub reads itself
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


def score_checkpoint(embeddings: Path, head: Path, tasks: Path, output: Path) -> list[str]:
    return run_clm_script([
        "evaluation/bon_eval.py", "--embeddings-dir", str(embeddings), "--checkpoint", str(head),
        "--tasks-file", str(tasks), "--n", str(settings.candidates), "--window", str(settings.window),
        "--output", str(output),
    ])


class DeepSWEJob:
    """One fine-tune and evaluation, written to <data_dir>/deepswe/<init>/seed_<seed>."""

    def __init__(self, init: str, seed: int):
        self.init = init
        self.seed = seed
        self.out_dir = deepswe_dir(config.data_dir) / init / f"seed_{seed}"
        self.heads_dir = download_authors_head(config.data_dir)
        self.held_out_file = self.heads_dir / "heldout_tasks.json"

    def run(self, compare_outcomes: bool = False) -> DeepSWEResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        init_checkpoint = self._init_checkpoint()
        with tempfile.TemporaryDirectory() as training_dir:
            samples, eval_dir = self._write_training_set(Path(training_dir))
            run_clm_script([
                "train/finetune.py", "--task", "clm", "--init-ckpt", str(init_checkpoint), "--out-dir", str(self.out_dir),
                "--emb-dir", training_dir, "--holdout-tasks", str(self.held_out_file), "--batch", str(settings.batch_size),
                "--seed", str(self.seed), "--epochs", str(settings.max_epochs), "--patience", str(settings.patience),
            ])
        volume.commit()
        if compare_outcomes:
            run_clm_script(["/root/outcome_ranking.py", str(self.out_dir), str(eval_dir), str(self.held_out_file), str(settings.window)])
        bon_lines = score_checkpoint(eval_dir, self.out_dir / "best_head.pt", self.held_out_file, self.out_dir / "bon.json")
        if compare_outcomes:
            score_checkpoint(eval_dir, self.out_dir / "ranked_head.pt", self.held_out_file, self.out_dir / "bon_ranked.json")
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

    def _write_training_set(self, training_dir: Path) -> tuple[list[DeepSWESample], Path]:
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
            torch.save(torch.load(eval_dir / name, weights_only=True, map_location="cpu")[rows_tensor], training_dir / name)
        metadata = {"num_samples": len(kept), "hidden_size": 4096, "samples": [sample.model_dump() for sample in kept]}
        (training_dir / "metadata.json").write_text(json.dumps(metadata))
        return kept, eval_dir


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], gpu="A10G", cpu=8, memory=64000, timeout=7200)
def train_and_evaluate(init: str, seed: int) -> DeepSWEResult:
    volume.reload()
    return DeepSWEJob(init, seed).run()


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], gpu="A10G",
              cpu=8, memory=32000, timeout=1200, retries=0, max_containers=1)
def matched_verifiers(init: str):
    """One sequential GPU allocation for the workshop's five pre-specified seeds."""
    volume.reload()
    for seed in (1234, 1, 2, 3, 4):
        job = DeepSWEJob(init, seed)
        if all((job.out_dir / f).is_file() for f in ("result.json", "bon.json", "bon_ranked.json", "outcome_selection.json")):
            continue
        job.run(compare_outcomes=True)


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], gpu="A10G",
              cpu=8, memory=32000, timeout=1200, retries=0, max_containers=1)
def evaluate_general(run: str):
    """Score the unchanged general checkpoint directly and under the workshop verifier protocol."""
    volume.reload()
    root = Path(config.data_dir)
    folder = root / config.runs_dir / run
    holdout = download_authors_head(config.data_dir) / "heldout_tasks.json"
    sys.path.insert(0, str(CLM_DIR / "preprocessing"))
    import hf_embeddings
    from finetuning.swe_outcomes import DATASET, REVISION

    embeddings = Path(hf_embeddings.download(settings.eval_dataset, str(deepswe_dir(config.data_dir) / "eval_embeddings")))
    for name, head in {"general": folder / "best.pt", "clm": root / "heads/qwen3_8b_posttrained_head.pt",
                       "mid": root / config.runs_dir / "mid/best.pt"}.items():
        score_checkpoint(embeddings, head, holdout, folder / f"deepswe_direct_{name}.json")
    # The publisher's reserved issues also measure cross-domain transfer of these same weights.
    cache = snapshot_download(DATASET, repo_type="dataset", revision=REVISION,
                              local_dir=root / "post_inputs" / REVISION, allow_patterns=["data/*.parquet"])
    with tempfile.TemporaryDirectory() as temp:
        source = Path(hf_embeddings.download(cache, temp))
        samples = json.loads((source / "metadata.json").read_text())["samples"]
        tasks = sorted({row["task_id"] for row in samples if row["config"] == "swebench_verified/val"})
        if len(tasks) != 95:
            raise ValueError("Expected the same 95 external test issues as the baseline")
        task_file = Path(temp) / "test_tasks.json"
        task_file.write_text(json.dumps(tasks))
        score_checkpoint(source, folder / "best.pt", task_file, folder / "swe_test_general.json")
    volume.commit()
    for seed in (1234, 1, 2, 3, 4):
        job = DeepSWEJob(run, seed)
        if not (job.out_dir / "result.json").exists():
            job.run()


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], gpu="A10G",
              cpu=6, memory=32000, timeout=1200, retries=0)
def train_swe_transfer(init: str):
    """Post-train a coding verifier on external SWE-bench rollouts, then test direct transfer."""
    volume.reload()
    root = Path(config.data_dir)
    source_head = root / config.runs_dir / init / "best.pt"
    out = root / config.runs_dir / f"post-swe-{uuid4().hex[:10]}"
    eval_dir = deepswe_dir(config.data_dir) / "eval_embeddings"
    if not (eval_dir / "metadata.json").exists():
        sys.path.insert(0, str(CLM_DIR / "preprocessing"))
        import hf_embeddings
        eval_dir = Path(hf_embeddings.download(settings.eval_dataset, str(eval_dir)))
    holdout = download_authors_head(config.data_dir) / "heldout_tasks.json"
    out.mkdir()
    run_clm_script(["/root/finetuning/swe_outcomes.py", str(root), str(source_head), str(out), str(eval_dir / "metadata.json")])
    volume.commit()
    score_checkpoint(eval_dir, out / "best.pt", holdout, out / "bon.json")
    volume.commit()
    print(f"Saved {out}", flush=True)


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], gpu="A10G",
              cpu=6, memory=32000, timeout=900, retries=0)
def swe_reference_scores(run: str):
    """Compare fixed checkpoints on the external data publisher's reserved SWE-bench tasks."""
    volume.reload()
    sys.path.insert(0, str(CLM_DIR / "preprocessing"))
    import hf_embeddings

    root = Path(config.data_dir)
    folder = root / config.runs_dir / run
    recipe = json.loads((folder / "transfer.json").read_text())
    heads = {
        "clm_post": root / "heads/qwen3_8b_posttrained_head.pt",
        "clm_deepswe": download_authors_head(config.data_dir) / "best_head.pt",
        "initial": Path(recipe["init"]), "specialist": folder / "best.pt",
    }
    with tempfile.TemporaryDirectory() as temp:
        source = Path(hf_embeddings.download(str(root / "post_inputs" / recipe["revision"]), temp))
        samples = json.loads((source / "metadata.json").read_text())["samples"]
        tasks = sorted({s["task_id"] for s in samples if s["config"] == "swebench_verified/val"})
        if len(tasks) != recipe["task_counts"]["test"]:
            raise ValueError("External test split disagrees with training record")
        task_file = Path(temp) / "test_tasks.json"
        task_file.write_text(json.dumps(tasks))
        for name, head in heads.items():
            score_checkpoint(source, head, task_file, folder / f"swe_test_{name}.json")
    volume.commit()


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


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], timeout=600)
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
