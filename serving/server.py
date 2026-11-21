"""Optional: serve the decision model over HTTP on a Modal GPU, for callers without a GPU of their own.
Qwen3-8B embeddings come from vLLM, and CLM's clm-serve scores them with the published heads.
To run the model on your own machine instead, use serving/decision_model.py.

    modal serve -m serving.server                     # prints the URL; Ctrl+C stops it
    python -m serving.decide --url URL                # one decision from this terminal

The URL is public, so the server only answers requests carrying the key in serving/game/session.json,
which is created on first use. The game page (serving/game/) and serving/client.py both send it.
"""
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

import modal
from huggingface_hub import hf_hub_download

from serving.client import session_key
from serving.settings import CONFIG_NAMES, PROJECT_DIR, clm, config, encoder_model

EMBEDDING_MODEL_NAME = "qwen3-8b"  # the name clm-serve asks vLLM for
WAIT_STEP_S = 2.0
settings = config.modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        f"git clone {clm.clm_repo_url} /clm && git -C /clm checkout {clm.clm_commit}",
        "pip install 'vllm==0.11.0' 'transformers>=4.56,<5' -e /clm",
    )
    .env({"HF_HOME": f"{settings.data_dir}/{settings.hf_cache_dir}"})
    .add_local_python_source("schemas", "serving")
)
for name in CONFIG_NAMES:
    image = image.add_local_file(PROJECT_DIR / name, f"/root/{name}")
volume = modal.Volume.from_name(settings.volume_name, version=2, create_if_missing=True)
# Only this machine knows the key; clm-serve reads it from CLM_API_KEY.
secrets = [modal.Secret.from_dict({"CLM_API_KEY": session_key() if modal.is_local() else ""})]
if settings.hf_secret_name:  # optional: Qwen3-8B and the heads are public
    secrets.append(modal.Secret.from_name(settings.hf_secret_name))
app = modal.App(settings.app_name)


def wait_until_up(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + settings.startup_timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{process.args[0]} exited with code {process.returncode}")
        try:
            with urlopen(url, timeout=WAIT_STEP_S) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(WAIT_STEP_S)
    raise TimeoutError(f"{url} did not come up")


def start_encoder() -> None:
    """vLLM in pooling mode: last-token embeddings, as in training."""
    encoder = settings.encoder
    process = subprocess.Popen([
        "vllm", "serve", encoder_model.model_repo, "--revision", encoder_model.model_revision,
        "--served-model-name", EMBEDDING_MODEL_NAME, "--runner", "pooling", "--enable-prefix-caching",
        "--max-model-len", str(encoder.max_model_len), "--gpu-memory-utilization", str(encoder.gpu_memory_utilization),
        "--host", "127.0.0.1", "--port", str(encoder.port), "--disable-uvicorn-access-log",
    ])
    wait_until_up(process, f"http://127.0.0.1:{encoder.port}/v1/models")


def start_heads(checkpoint: str) -> None:
    """clm-serve projects the embeddings with the heads; --cors lets the game page call it."""
    subprocess.Popen([
        "clm-serve", "--host", "0.0.0.0", "--port", str(settings.heads_port), "--ckpt", checkpoint,
        "--emb-url", f"http://127.0.0.1:{settings.encoder.port}/v1/embeddings", "--emb-model", EMBEDDING_MODEL_NAME,
        "--no-ui", "--no-download", "--cors",
    ])


@app.function(image=image, gpu=settings.gpu, volumes={settings.data_dir: volume}, secrets=secrets,
              scaledown_window=settings.scaledown_window_s, max_containers=1)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=settings.heads_port, startup_timeout=settings.startup_timeout_s)
def serve() -> None:
    start_encoder()
    heads = config.heads
    checkpoint = hf_hub_download(heads.hf_repo, heads.hf_file, revision=heads.hf_revision)  # cached in HF_HOME
    volume.commit()  # keeps the Qwen3-8B and checkpoint downloads for the next start
    start_heads(checkpoint)
