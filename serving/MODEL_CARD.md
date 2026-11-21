---
license: apache-2.0
base_model: Qwen/Qwen3-8B
pipeline_tag: text-ranking
language:
- en
tags:
- contrastive-learning
- verifier
- reranker
- agents
- clm
library_name: contrastive-lm
---

<h3 align="center">NovaCAPE-Qwen3-8B</h3>

<p align="center">
| 📄 <b>Blog</b> (TODO: link the series once published)
| 💻 <a href="https://github.com/MKDYasserH/system-one-model"><b>Code</b></a>
| 🧠 Based on <a href="https://huggingface.co/Contrastive-LM/CLM-v0.1-8B"><b>CLM-8B</b></a> |
</p>

**NovaCAPE** is a **System One (decision) model**: given a state and a few options written as text, it returns a
probability for each option in a single pass, without generating text. It is a from-scratch reproduction of
[CLM-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B): two small projection heads (a state head and an action
head) on top of a frozen **Qwen3-8B** encoder, trained with a bidirectional InfoNCE loss. It uses the same
format and serving code as CLM-8B, so it is a drop-in replacement.

- **Training:** pre-trained on 62.45M Nemotron Q&A pairs, mid-trained on 26.7M synthetic hard negatives, then
  post-trained with a general recipe: terminal agent steps, choices within agent environments, Q&A replay, and
  answer preferences with critiques.
- **Tricky Q&A:** 70.8% top-1 against up to 10 hard negatives (CLM-8B: 68.6%).
- **Fine-tuned as a verifier:** 30.4 / 38 DeepSWE tasks resolved, best of 4 (CLM-8B: 31.0).
- **Fast and cheap to adapt:** only the heads (18.9M parameters) are trained, on frozen embeddings.

## Usage

### With the `contrastive-lm` package

Exactly as for CLM-8B, with this checkpoint passed to `clm-serve`:

```bash
pip install "contrastive-lm @ git+https://github.com/Contrastive-LM/CLM.git@d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"

# 1. encoder (Qwen3-8B embeddings, at the revision the heads were trained on)
vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --served-model-name qwen3-8b \
    --runner pooling --max-model-len 2048 --port 8090 &

# 2. API + playground at http://localhost:8700/, serving NovaCAPE
clm-serve --ckpt "$(hf download mkdyasserh/NovaCAPE-Qwen3-8B best.pt)"
```

Ask typed questions about a state:

```python
from clm import CLMClient, Choice

client = CLMClient()  # http://127.0.0.1:8700 by default
r = client.system_one(
    state="The summer I learned to ride a bike at my grandparents' house.",
    questions={
        "memory": Choice(instructions="Which kind of memory is this?",
                         criteria={"childhood": "A memory from childhood: being a kid, school, playing, growing up.",
                                   "travel": "A memory of a trip: travelling, discovering a new place.",
                                   "work": "A memory from working life: a job, colleagues, a career.",
                                   "love": "A memory of a romantic relationship."}),
    },
)
print(r.answers["memory"].choice)         # childhood
print(r.answers["memory"].probabilities)  # childhood 0.54, work 0.27, love 0.14, travel 0.04
```

`Noul` (yes/no) and `Score` (ordered levels) questions, and the `/v1/rank` endpoint, work as described in the
[CLM documentation](https://github.com/Contrastive-LM/CLM).

### Without vLLM (macOS, CPU, any CUDA machine)

vLLM needs Linux and a CUDA GPU. The [code repository](https://github.com/MKDYasserH/system-one-model) runs the
same model with PyTorch and `transformers`, and matches the `clm-serve` numbers to within 0.005:

```bash
git clone https://github.com/MKDYasserH/system-one-model && cd system-one-model && uv sync
python -m serving.decide --state "The summer I learned to ride a bike at my grandparents' house."
```

It needs about 17 GB of memory for Qwen3-8B in bfloat16 (a 24 GB GPU or Apple Silicon Mac).

### Fine-tuning

Only the heads are trained, so fine-tuning is cheap. With CLM's script, starting from this checkpoint:

```bash
git clone https://github.com/Contrastive-LM/CLM.git && cd CLM && pip install -e .
hf download Contrastive-LM/deepswe-clm-heads-8k heldout_tasks.json --local-dir heads/deepswe
python train/finetune.py --task clm --init-ckpt "$(hf download mkdyasserh/NovaCAPE-Qwen3-8B best.pt)" \
    --out-dir runs/deepswe --holdout-tasks heads/deepswe/heldout_tasks.json --batch 512
```

## Results

| Test | NovaCAPE | CLM-8B | Random | Perfect |
|---|---:|---:|---:|---:|
| Tricky Q&A, top-1 among up to 10 hard negatives (191,161 held-out questions) | **70.8%** | 68.6% | ≈ 10% | 100% |
| DeepSWE, tasks resolved out of 38, best of 4, after verifier fine-tuning (5 seeds) | 30.4 | **31.0** | 28 | 34 |
| SWE-bench Verified, expected best-of-4 success on 95 issues, zero-shot | **71.3%** | 69.4% | 70.7% | 83.3% |
| T-Rex game, agreement with the planner (5 fresh courses, shield on) | 36.1% | **66.9%** | | |

## Limitations

- **Encoder-locked:** the heads require Qwen3-8B (revision `b968826d`) last-token-pooled embeddings.
- **No generation:** NovaCAPE only scores the candidates you give it, and its probabilities are relative to that set.
- **Sensitive to wording in zero-shot use:** describe each option in a short phrase, and ask what a text *is about*
  rather than what to do with it. Check on a few labeled examples before relying on it.
- **No numeric reasoning:** it matches meanings, not quantities; states described with distances or speeds call
  for fine-tuning.
- **Verifier results need fine-tuning:** the DeepSWE number comes from fine-tuned heads, not this checkpoint
  zero-shot.

## Citation

NovaCAPE reproduces CLM; please cite the original work:

```bibtex
@misc{kwok2026contrastivelanguagemodels,
  title={Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making},
  author={Jacky Kwok and Hangoo Kang and Tarun Suresh and Jon Saad-Falcon and Marco Pavone and Christopher Ré and Azalia Mirhoseini},
  year={2026},
  note={Notion Blog},
  url={https://contrastive-lm.notion.site}
}
```

## License

The NovaCAPE weights are released under the Apache 2.0 License. The base encoder
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) is also Apache 2.0.
