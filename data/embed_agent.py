"""Embed the agent (state, action) pairs with frozen Qwen3-8B, the same way the Q&A data was embedded:
last-token pooling, L2 normalized, float16. States keep their last max_tokens tokens, actions their first.

Reads <data_dir>/agent/pairs/<subset>.parquet (from load_agent_data.py) and writes, row for row:
    <data_dir>/agent/vec/<subset>_q.npy, _a.npy (N, 4096) float16 and _meta.parquet (subset, traj, step, split)

Needs vLLM on a GPU, so it only runs on Modal (run_mode = "remote"):
    modal run --detach data/embed_agent.py
"""
import os
import tomllib
from itertools import batched
from pathlib import Path
from time import time

import modal
import numpy as np
import pyarrow.parquet as pq
from pydantic import TypeAdapter

from schemas.agent_data import ChatMessage, EmbeddingReport
from schemas.data_loader import Config

# Modal mounts this script at /root, so the config sits next to it in the container too.
CONFIG_PATH = Path(__file__).parent / "config.toml"

with CONFIG_PATH.open("rb") as config_file:
    config = Config.model_validate(tomllib.load(config_file))

settings = config.agent_embedding
pairs_dir = Path(config.data_dir) / config.agent_pairs.output_dir
vectors_dir = Path(config.data_dir) / settings.output_dir
model_dir = Path(config.data_dir) / settings.model_dir
META_COLUMNS = ["subset", "traj", "step", "split"]  # what post-training reads; the text stays in agent/pairs

# vLLM pins its own torch, so this image doesn't reuse the project's uv environment.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers>=4.56,<5", "pyarrow==18.1.0", "huggingface_hub", "pydantic>=2.10")
    .env({"HF_HOME": f"{config.data_dir}/hf"})
    .add_local_file(CONFIG_PATH, f"/root/{CONFIG_PATH.name}")
    .add_local_python_source("schemas")
)
volume = modal.Volume.from_name(config.volume_name, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name(config.hf_secret_name)  # sets HF_TOKEN, which huggingface_hub reads itself
app = modal.App(settings.app_name)

chat_messages = TypeAdapter(list[ChatMessage])


def recent_messages(messages: list[ChatMessage], char_budget: int) -> list[ChatMessage]:
    """The newest messages holding about char_budget characters, so we never tokenize a huge history
    only to keep its end. The message that crosses the budget is kept; the exact cut is by tokens."""
    kept, total = [], 0
    for message in reversed(messages):
        kept.append(message)
        total += len(message.content)
        if total > char_budget:
            break
    return kept[::-1]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is a cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / (norms + 1e-12)).astype(np.float16)


def save_array(path: Path, array: np.ndarray) -> None:
    partial = path.with_name(path.name + ".part")
    with partial.open("wb") as file:
        np.save(file, array)
    os.replace(partial, path)


@app.cls(image=image, volumes={config.data_dir: volume}, gpu="L40S", cpu=8, memory=48000, timeout=7200, max_containers=8, scaledown_window=120)
class AgentEmbedder:
    @modal.enter()
    def load_model(self):
        """Runs once per container, so every subset it embeds reuses the loaded model."""
        from transformers import AutoTokenizer
        from vllm import LLM

        volume.reload()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # "pooling" runs one forward pass per text and returns a vector instead of generating.
        self.llm = LLM(
            model=str(model_dir), runner="pooling", max_model_len=settings.max_model_len,
            gpu_memory_utilization=settings.gpu_memory_utilization,
        )

    @modal.method()
    def embed_subset(self, subset: str) -> EmbeddingReport:
        volume.reload()
        meta_path = vectors_dir / f"{subset}_meta.parquet"
        if meta_path.exists():
            return EmbeddingReport(subset=subset, pairs=pq.read_metadata(meta_path).num_rows, already_embedded=True)
        start_time = time()
        table = pq.read_table(pairs_dir / f"{subset}.parquet")
        state_tokens = [self._state_tokens(chat_messages.validate_json(state)) for state in table.column("state").to_pylist()]
        action_tokens = [self._action_tokens(action) for action in table.column("action").to_pylist()]

        vectors_dir.mkdir(parents=True, exist_ok=True)
        save_array(vectors_dir / f"{subset}_q.npy", self._embed(state_tokens))
        save_array(vectors_dir / f"{subset}_a.npy", self._embed(action_tokens))
        # Written last: its presence means the subset is done, which is what makes a rerun resume.
        pq.write_table(table.select(META_COLUMNS), meta_path)
        volume.commit()

        report = EmbeddingReport(
            subset=subset, pairs=table.num_rows, tokens=sum(map(len, state_tokens)) + sum(map(len, action_tokens)),
            minutes=(time() - start_time) / 60,
        )
        print(report, flush=True)
        return report

    def _state_tokens(self, state: list[ChatMessage]) -> list[int]:
        """The state in Qwen's chat format (<|im_start|>role ... <|im_end|>), keeping its newest tokens."""
        from jinja2 import TemplateError

        messages = [message.model_dump() for message in recent_messages(state, settings.state_char_budget)]
        try:
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
        except TemplateError:
            # Qwen's template refuses some role orders (e.g. no user turn left after the cut): embed plain text.
            tokens = self.tokenizer("\n\n".join(m["content"] for m in messages), add_special_tokens=False)["input_ids"]
        return list(tokens)[-settings.max_tokens:]

    def _action_tokens(self, action: str) -> list[int]:
        """Plain text, like the Q&A answers. The start of an action says what the agent does."""
        return self.tokenizer(action, add_special_tokens=False)["input_ids"][: settings.max_tokens]

    def _embed(self, token_lists: list[list[int]]) -> np.ndarray:
        """(len(token_lists), 4096). Texts go to vLLM texts_per_call at a time to bound memory."""
        from vllm.inputs import TokensPrompt

        parts = []
        for texts in batched(token_lists, settings.texts_per_call):
            # TokensPrompt: already tokenized, so vLLM embeds exactly the tokens we kept.
            outputs = self.llm.embed([TokensPrompt(prompt_token_ids=tokens) for tokens in texts], use_tqdm=False)
            parts.append(np.stack([np.asarray(output.outputs.embedding, dtype=np.float32) for output in outputs]))
        return l2_normalize(np.concatenate(parts))


def download_model() -> None:
    """Pinned to a commit so the embeddings can't change if the model repo is updated."""
    from huggingface_hub import snapshot_download

    volume.reload()
    snapshot_download(
        settings.model_repo, revision=settings.model_revision, local_dir=model_dir,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    volume.commit()


@app.function(image=image, volumes={config.data_dir: volume}, secrets=[hf_secret], timeout=14000)
def orchestrate():
    download_model()
    subsets = sorted(path.stem for path in pairs_dir.glob("*.parquet"))
    print(f"{len(subsets)} subsets", flush=True)
    total = 0
    for result in AgentEmbedder().embed_subset.map(subsets, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            print("subset failed", repr(result)[:300], flush=True)
            continue
        total += result.pairs
        print(f"{result.subset}: {result.pairs} pairs, {total} in total", flush=True)
    print(f"DONE {total} pairs", flush=True)


@app.local_entrypoint()
def main():
    if config.run_mode == "local":
        raise SystemExit('embedding needs vLLM on a GPU: set run_mode = "remote" in data/config.toml')
    orchestrate.remote()
