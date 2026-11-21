"""Shared pinned Qwen embedding service for general data preparation and evaluation."""
import subprocess
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import modal

HERE = Path(__file__).parent
with (HERE.parent / "evaluation/config.toml").open("rb") as stream:
    config = tomllib.load(stream)
with (HERE / "config.toml").open("rb") as stream:
    encoder = tomllib.load(stream)["agent_embedding"]

volume = modal.Volume.from_name(config["volume_name"], version=2)
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("git")
    .run_commands(f"git clone {config['deepswe']['clm_repo_url']} /clm && git -C /clm checkout {config['deepswe']['clm_commit']}",
                  "pip install 'vllm==0.11.0' 'transformers>=4.56,<5' -e /clm 'httpx==0.28.1'")
    .add_local_file(HERE.parent / "evaluation/config.toml", "/root/evaluation/config.toml")
    .add_local_file(HERE / "config.toml", "/root/data/config.toml")
    .add_local_file(HERE / "qwen_service.py", "/root/data/qwen_service.py")
)


@contextmanager
def service(command, url):
    process = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{command[0]} exited with {process.returncode}")
            try:
                with urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError(url)
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def encoder_service():
    return service([
        "vllm", "serve", encoder["model_repo"], "--revision", encoder["model_revision"],
        "--host", "127.0.0.1", "--served-model-name", "qwen3-8b", "--runner", "pooling",
        "--enable-prefix-caching", "--max-model-len", "2048", "--port", "8090",
        "--gpu-memory-utilization", "0.6", "--disable-uvicorn-access-log",
    ], "http://127.0.0.1:8090/v1/models")


resources = dict(image=image, gpu="L40S", cpu=8, memory=48000, volumes={config["data_dir"]: volume}, retries=0)
root = Path(config["data_dir"])
