import os
import shutil
import tempfile
import tomllib
import modal

from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from time import time
from huggingface_hub import HfApi, hf_hub_download
from schemas.data_loader import Config, Job, RemoteFile, Source


PROJECT_DIR = Path(__file__).parent.parent
# Modal mounts this script at /root, so the config sits next to it in the container too.
CONFIG_PATH = Path(__file__).parent / "config.toml"

with CONFIG_PATH.open("rb") as config_file:
    config = Config.model_validate(tomllib.load(config_file))

IS_LOCAL = config.run_mode == "local"
data_dir = config.local.data_dir if IS_LOCAL else config.data_dir

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(str(PROJECT_DIR))
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(CONFIG_PATH, f"/root/{CONFIG_PATH.name}")
    .add_local_python_source("schemas")
)
volume = modal.Volume.from_name(config.volume_name, create_if_missing=True, version=2)
app = modal.App(config.app_name)


def download_file(job: Job, file: RemoteFile) -> None:
    destination = job.destination(data_dir, file)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = hf_hub_download(
            repo_id=job.source.repo, filename=file.path, repo_type="dataset", revision=job.source.revision, local_dir=temp_dir
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(path, destination + ".part")
        os.replace(destination + ".part", destination)


@app.function(image=image, volumes={config.data_dir: volume}, cpu=4, memory=16384, timeout=3600, retries=modal.Retries(max_retries=6, initial_delay=30.0), max_containers=16)
def fetch(job: Job) -> int:
    done = 0
    for file in job.files:
        destination = job.destination(data_dir, file)
        if os.path.exists(destination) and os.path.getsize(destination) == file.size:
            continue
        download_file(job, file)
        done += file.size
    if not IS_LOCAL:
        volume.commit()
    return done


def list_files(source: Source) -> list[RemoteFile]:
    entries = HfApi().list_repo_tree(source.repo, repo_type="dataset", revision=source.revision, recursive=True)
    return [
        RemoteFile(path=entry.path, size=entry.size)
        for entry in entries
        if hasattr(entry, "size") and any(entry.path.endswith(ext) for ext in source.extensions)
    ]


def chunk_key(file: RemoteFile) -> str:
    # shard_05/chunk_0073_q.npy -> "shard_05/0073"
    folder, filename = file.path.split("/")[0], file.path.split("/")[-1]
    return f"{folder}/{filename.split('_')[1]}"


def group_by_chunk(files: list[RemoteFile]) -> list[list[RemoteFile]]:
    ordered = sorted(files, key=chunk_key)
    return [list(group) for _, group in groupby(ordered, key=chunk_key)]


def print_summary(label: str, files: list[RemoteFile]) -> None:
    print(f"{label}: {len(files)} files, {sum(f.size for f in files) / 1e12:.3f} TB", flush=True)


def build_jobs(source: Source) -> list[Job]:
    files = list_files(source)
    print_summary(source.kind, files)
    groups = group_by_chunk(files) if source.group_by_chunk else [[file] for file in files]
    jobs = [Job(source=source, files=group) for group in groups]
    return jobs[: config.local.jobs_per_source] if IS_LOCAL else jobs


def fetch_locally(jobs: list[Job]) -> Iterator[int | Exception]:
    # Mirrors fetch.map(return_exceptions=True): a failed job is reported, not raised.
    for job in jobs:
        try:
            yield fetch.local(job)
        except Exception as error:
            yield error


def run_jobs(jobs: list[Job]) -> int:
    start_time = time()
    total = 0
    results = fetch_locally(jobs) if IS_LOCAL else fetch.map(jobs, order_outputs=False, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print("job failed", repr(result)[:200], flush=True)
            continue
        total += result
        if i % config.log_every == 0:
            print(f"{i}/{len(jobs)} jobs, {total / 1e9:.0f} GB new, {(time() - start_time) / 60:.1f} min", flush=True)
    return total


def download_heads() -> None:
    if not IS_LOCAL:
        volume.reload()
    heads_dir = os.path.join(data_dir, "heads")
    os.makedirs(heads_dir, exist_ok=True)
    for filename in config.heads.files:
        shutil.copyfile(hf_hub_download(config.heads.repo, filename), os.path.join(heads_dir, filename))
    if not IS_LOCAL:
        volume.commit()


@app.function(image=image, volumes={config.data_dir: volume}, timeout=86400)
def orchestrate():
    jobs = [job for source in config.sources for job in build_jobs(source)]

    start_time = time()
    total = run_jobs(jobs)
    download_heads()
    print(f"DONE {total / 1e9:.0f} GB new in {(time() - start_time) / 60:.1f} min", flush=True)


@app.local_entrypoint()
def main():
    if IS_LOCAL:
        orchestrate.local()
    else:
        orchestrate.remote()
