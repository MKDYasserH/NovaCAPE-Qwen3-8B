"""Run the decision model wherever PyTorch runs (CUDA, an Apple GPU or a CPU): Qwen3-8B embeds the texts,
the published heads score them.

    from serving.decision_model import DecisionModel
    model = DecisionModel.load()                    # the first call downloads Qwen3-8B (16.4 GB) and the heads
    decision = model.choose(state, question)         # one decision
    decisions = model.classify(states, question)     # many texts, the options embedded once

Texts are laid out as CLM's serving engine lays them out, which is how the series evaluated the heads: the state
head reads the state, a blank line and the instructions; the action head reads each option's description.
"""
import time

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from finetuning.devices import pick_device
from finetuning.heads import ContrastiveHeads
from schemas.serving import ChoiceQuestion, Decision, LocalModelSettings, OptionProbability
from serving.settings import config, encoder_model, training


def state_text(state: str, instructions: str) -> str:
    """Context first, question last, as in CLM's schema.state_text."""
    state, instructions = state.strip(), instructions.strip()
    return f"{state}\n\n{instructions}" if state and instructions else state or instructions


class QwenEncoder:
    """Last-token embeddings of the frozen Qwen3-8B, L2-normalized, as vLLM's pooling runner returns them."""

    def __init__(self, settings: LocalModelSettings, device: torch.device):
        self.settings = settings
        self.device = device
        # Left padding keeps every text's last token at the end; left truncation keeps a long text's end.
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_model.model_repo, revision=encoder_model.model_revision, padding_side="left", truncation_side="left")
        self.model = AutoModel.from_pretrained(  # the base model: no language-model head, about 1.2 GB less
            encoder_model.model_repo, revision=encoder_model.model_revision, dtype=getattr(torch, settings.dtype))
        self.model.to(device).eval()

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> torch.Tensor:
        """[len(texts), 4096] float32 on the encoder's device."""
        parts = []
        for start in range(0, len(texts), self.settings.batch_size):
            batch = self.tokenizer(texts[start:start + self.settings.batch_size], padding=True, truncation=True,
                                   max_length=self.settings.max_tokens, return_tensors="pt").to(self.device)
            positions = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)  # padding doesn't shift positions
            hidden = self.model(**batch, position_ids=positions).last_hidden_state[:, -1]
            parts.append(F.normalize(hidden.float(), dim=-1))
        return torch.cat(parts)


def load_heads(device: torch.device) -> ContrastiveHeads:
    source = config.heads
    path = hf_hub_download(source.hf_repo, source.hf_file, revision=source.hf_revision)
    heads = ContrastiveHeads(training.head, training.loss)
    heads.load_checkpoint_weights(torch.load(path, map_location="cpu", weights_only=True))
    return heads.to(device).eval()


class DecisionModel:
    def __init__(self, encoder: QwenEncoder, heads: ContrastiveHeads):
        self.encoder = encoder
        self.heads = heads
        self.option_cache: dict[str, torch.Tensor] = {}  # option text -> its vector, so repeated options cost nothing

    @classmethod
    def load(cls) -> "DecisionModel":
        settings = config.local
        device = pick_device() if settings.device == "auto" else torch.device(settings.device)
        return cls(QwenEncoder(settings, device), load_heads(device))

    @torch.inference_mode()
    def classify(self, states: list[str], question: ChoiceQuestion) -> list[Decision]:
        """One decision per state, all against the same options."""
        started = time.perf_counter()
        state_vectors = self.heads.embed_states(self.encoder.embed([state_text(s, question.instructions) for s in states]))
        option_vectors = self.option_vectors([option.description for option in question.options])
        probabilities = (self.heads.logit_scale() * state_vectors @ option_vectors.T).softmax(dim=-1).tolist()
        latency_ms = (time.perf_counter() - started) * 1000 / len(states)
        return [
            Decision(probabilities=[OptionProbability(name=o.name, probability=p) for o, p in zip(question.options, row)],
                     latency_ms=latency_ms)
            for row in probabilities
        ]

    def choose(self, state: str, question: ChoiceQuestion) -> Decision:
        return self.classify([state], question)[0]

    def option_vectors(self, descriptions: list[str]) -> torch.Tensor:
        missing = [d for d in dict.fromkeys(descriptions) if d not in self.option_cache]
        if missing:
            for description, vector in zip(missing, self.heads.embed_actions(self.encoder.embed(missing))):
                self.option_cache[description] = vector
        return torch.stack([self.option_cache[d] for d in descriptions])
