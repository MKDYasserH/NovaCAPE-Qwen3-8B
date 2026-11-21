"""Run the pinned CLM real-time T-Rex harness on the selected general post-training head."""
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import modal

from data.qwen_service import config, encoder_service, resources as encoder_resources, root, service, volume

HERE = Path(__file__).parent
app = modal.App("nova-trex")
resources = {**encoder_resources,
             "image": encoder_resources["image"].add_local_file(
                 HERE.parent / "finetuning/trex_choices.py", "/root/finetuning/trex_choices.py"
             )}


def head_service(checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return service(["clm-serve", "--no-ui", "--host", "127.0.0.1", "--ckpt", str(checkpoint)],
                   "http://127.0.0.1:8700/health")


def run_courses(result: Path, seed: int, shield: bool):
    subprocess.run([sys.executable, "/clm/examples/t_rex/run.py", "--model", "clm", "--seeds", "5",
                    "--seed", str(seed), "--out", str(result), *([] if shield else ["--no-shield"])],
                   check=True, timeout=480)
    report = json.loads(result.read_text())
    if [r["seed"] for r in report["results"]] != list(range(seed, seed + 5)) or any(
        row["errors"] or not row["decisions"] for row in report["results"]
    ):
        raise RuntimeError(f"Incomplete game evaluation: {report['summary']}")
    volume.commit()
    return report["summary"]


@app.function(**resources, timeout=1800)
def play(run: str):
    volume.reload()
    folder = root / config["runs_dir"] / run
    with encoder_service(), head_service(folder / "best.pt"):
        for name, shield in (("shield", True), ("no_shield", False)):
            run_courses(folder / f"trex_{name}.json", 0, shield)


@app.function(**resources, timeout=1100)
def measure_agreement(run: str, seed: int, shield: bool = True):
    volume.reload()
    if run == "released-clm":
        head = root / "heads/qwen3_8b_posttrained_head.pt"
        folder = root / "trex" / run
    else:
        folder = root / config["runs_dir"] / run
        head = folder / "best.pt"
    folder.mkdir(exist_ok=True, parents=True)
    with encoder_service(), head_service(head):
        name = "shield" if shield else "no_shield"
        return run_courses(folder / f"{name}_{seed}.json", seed, shield)


@app.function(**resources, timeout=1000)
def fit_game_head(parent: str):
    volume.reload()
    runs = root / config["runs_dir"]
    checkpoint = runs / parent / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    folder = runs / f"post-trex-{uuid4().hex[:10]}"
    folder.mkdir()
    with encoder_service():
        subprocess.run([sys.executable, "/root/finetuning/trex_choices.py", str(checkpoint), str(folder)],
                       check=True, timeout=360)
    volume.commit()
    print(f"Locked {folder}", flush=True)
    return folder.name


@app.local_entrypoint()
def specialize(parent: str = "post-terminal-870f73d383"):
    run = fit_game_head.remote(parent)
    for summary in measure_agreement.starmap([(run, 0, True), (run, 7000, True), (run, 0, False)]):
        print(run, summary, flush=True)


@app.local_entrypoint()
def benchmark_general(run: str):
    """Evaluate one locked head on both course sets and on the unshielded workshop courses."""
    for summary in measure_agreement.starmap([(run, 0, True), (run, 7000, True), (run, 0, False)]):
        print(run, summary, flush=True)
