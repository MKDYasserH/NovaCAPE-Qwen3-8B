# System One Model from Scratch

NovaCAPE-Qwen3-8B is a decision model based on [CLM-8B](https://contrastive-lm.notion.site).

A System One model does not write text. You give it a situation and a list of options, and it returns a probability for each option in one pass. [Jev](https://typesafe.ai) made the idea popular but is closed source, so this repo builds an open equivalent.

**Site with games and results:** https://system-one-workshop.vercel.app

[General post-training from mid, measured comparisons and reproduction commands](evaluation/results/GENERAL_POST_TRAINING.md)

## The model

```text
Question + document -> Qwen3-8B (frozen) -> State head  (9.44M) -> 512-d
Answer              -> Qwen3-8B (frozen) -> Action head (9.44M) -> 512-d
```

We take the hidden state of the last token (4096 numbers), normalize it, and pass it through a small network: 4096 -> 1536 -> 1536 (LayerNorm) -> 512, with GELU. The score is the cosine of the two vectors times a learned scale. Only the two heads are trained, about 18.9M parameters.

## Use the model

The trained heads are on Hugging Face: [mkdyasserh/NovaCAPE-Qwen3-8B](https://huggingface.co/mkdyasserh/NovaCAPE-Qwen3-8B). Every decision also runs the frozen [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), at the revision the heads were trained on (`b968826d`), which needs about 17 GB of memory in bfloat16: a 24 GB GPU, an Apple Silicon Mac with 24 GB or more, or a CPU with enough RAM (slow).

### On your machine: CUDA, Apple GPU or CPU

`serving/decision_model.py` runs Qwen3-8B with PyTorch and `transformers`, and scores with the published heads. The first run downloads Qwen3-8B (16.4 GB) and the heads (75 MB) into your Hugging Face cache.

```bash
uv sync
uv run python -m serving.decide                   # an example: which kind of memory is this?
uv run python -m serving.decide --state "The export button crashes the whole page." \
    --instructions "What is this message about?" \
    --option billing="Payments, invoices, refunds, subscription charges." \
    --option technical="Bugs, errors, crashes, login problems." \
    --option account="Changing email, password resets, deleting an account."
```

From Python, `choose` makes one decision and `classify` scores many texts against the same options, embedding the options once:

```python
from schemas.serving import ChoiceQuestion, Option
from serving.decision_model import DecisionModel

model = DecisionModel.load()
question = ChoiceQuestion(
    instructions="Which kind of memory is this?",
    options=[
        Option(name="childhood", description="A memory from childhood: being a kid, school, playing, growing up."),
        Option(name="travel", description="A memory of a trip: travelling, discovering a new place."),
    ],
)
decision = model.choose("The summer I learned to ride a bike at my grandparents' house.", question)
print(decision.chosen, decision.probabilities)
decisions = model.classify(["Our honeymoon in Lisbon.", "My first day at primary school."], question)
```

On an Apple M4 Pro with 24 GB, loading takes about 11 s, one decision about 125 ms, and batches about 60 ms per text. The numbers match CLM's GPU server (below) to within 0.005.

### How to ask

The state head reads the state, a blank line, then the instructions; the action head reads each option's description, unchanged (the same layout as CLM's server). In zero-shot use the wording matters:

- Describe each option in a short phrase, not a bare label.
- Ask what the text is about rather than what to do with it: on eight support tickets, "What is this message about?" sorted 5 correctly, while "Which team should handle this ticket?" sent all eight to one team.
- Check on a few dozen labeled examples before relying on it. If zero-shot is not enough, fine-tune the heads: Qwen3-8B stays frozen, so it is cheap.

### With CLM's server: Linux and a CUDA GPU

For high throughput, serve Qwen3-8B with vLLM and the heads with CLM's own `clm-serve`, which also has a web playground:

```bash
pip install "vllm==0.11.0" "contrastive-lm @ git+https://github.com/Contrastive-LM/CLM.git@d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --served-model-name qwen3-8b \
    --runner pooling --max-model-len 2048 --port 8090 &
clm-serve --ckpt "$(hf download mkdyasserh/NovaCAPE-Qwen3-8B best.pt)"
uv run python -m serving.decide --url http://127.0.0.1:8700    # or CLM's own CLMClient
```

### On Modal: no GPU of your own (optional)

`serving/server.py` runs the same two servers on a Modal GPU and prints a public URL, protected by a key kept in `serving/game/session.json` (created on first use, git-ignored). The GPU shuts down 5 minutes after the last request.

```bash
uv run modal serve -m serving.server                    # Ctrl+C stops it
uv run python -m serving.decide --url URL
```

Settings for all three are in `serving/config.toml`: where the heads come from, and the device and precision for local runs.

## What was public, and what we rebuilt

The CLM team released the CLM-8B weights, described all three training stages, and published a trainer for fine-tuning the projection heads on new data, including DeepSWE traces. The official repository does not include the end-to-end scripts used to pre-train, mid-train and post-train the released CLM-8B checkpoint. [CLM release](https://github.com/Contrastive-LM/CLM/tree/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6) · [Fine-tuning script](https://github.com/Contrastive-LM/CLM/blob/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6/train/finetune.py)

The pre-training and mid-training recipes use settings from the authors’ checkpoints. [CLM head checkpoints](https://huggingface.co/jackyk02/qwen3_8b_head)

| Stage | Data | Recipe | Script |
|---|---|---|---|
| Vectors | [62.45M Q&A vectors](https://huggingface.co/datasets/Contrastive-LM/CLM-v0.1-Pretrain-Nemotron), about 1 TB; [hard negatives](https://huggingface.co/datasets/jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings), about 0.25 TB | Copy to a Modal volume | `data/load_data.py` |
| Pre-training | 621 chunks | Batch 16,384, lr 0.00653 constant, AdamW, no weight decay, temperature starts at 0.07 and is learned (scale capped at 100), same-document pairs masked | `finetuning/pre_training_handler.py` |
| Mid-training | 2.8M pairs with up to 10 hard negatives each | Batch 16,384, lr 0.001 constant, negatives added to the state-to-action direction, weight 1.0 | `finetuning/mid_training_handler.py` |
| Post-training | 847k agent decisions from [ADP v2](https://huggingface.co/datasets/neulab/adp-v2) | **Our reconstruction.** 60% agent steps, 40% replay with hard negatives, lr 0.0003, 400 steps, temperature frozen | `data/load_agent_data.py`, `data/embed_agent.py`, `finetuning/post_training_handler.py` |
| Verifier | 22,576 passing DeepSWE steps | Batch 512, seed 1234, the repo's own `finetune.py` | `evaluation/deepswe.py` |
| Testing | Held-out `shard_00/0028-0029` and `shard_05/0083-0084` | One code path for every head | not ported yet |

## Run it

You need a [Modal](https://modal.com) account and about 1.3 TB of volume storage. No API keys are needed for the public data. Everything goes to one Modal volume, `nova-data`, mounted at `/data`.

### Low-cost post-training from an existing mid checkpoint

Run the terminal-agent sweep directly after mid-training. It reads `runs/mid/best.pt` and the existing Q&A hard negatives. A CPU job stages 5 GB of published, pinned Qwen3-8B embeddings. One L4 then runs three 120-step trials with identical batches. The original ADP recipe is still available in the full pipeline below. [Terminal-agent data card](https://huggingface.co/datasets/Hkang/litecoder-terminal-sft-qwen3-8b-embeddings)

```bash
uv run modal run -m finetuning.post_sweep::run
```

The command prints a fresh `post-terminal-…` run name. Its directory contains the selected `best.pt` and the full comparison in `selection.json`. The choices are two-head contrastive training, an additional loss that matches two distant actions to their own states, and the same additional loss with a frozen action head. The sweep selects the highest retrieval accuracy on fixed terminal validation rows, subject to Q&A accuracy staying within 1.5 percentage points of mid on a separate replay chunk. The original mid head is eligible. The official Q&A chunks and DeepSWE tasks are reserved for evaluation.

Use the printed run name in these commands. The first command scores all 191,161 tricky Q&A rows. The second runs the published five DeepSWE seeds and records both the conventional verifier and an experimental outcome-adjusted verifier. The third evaluates the general head in the upstream five-course T-Rex game.

```bash
uv run modal run -m finetuning.post_sweep::final_qa --run RUN_NAME
uv run modal run evaluation/deepswe.py::matched_verifiers --init RUN_NAME
uv run modal run -m evaluation.trex::play --run RUN_NAME
```

For a coding-specific head, train on pairs of passing and failing attempts at the same SWE-bench issue. Training uses the mean cosine score over each attempt's final 12 steps. Selection uses external SWE-bench development issues. The job saves a new `post-swe-…/best.pt`, writes the external test comparison to `transfer.json`, and measures direct transfer to DeepSWE in `bon.json`. The original terminal head stays available. [Coding rollout data card](https://huggingface.co/datasets/tarsur385/swebench-verified-top15-embeddings-8k)

```bash
uv run modal run evaluation/deepswe.py::train_swe_transfer --init RUN_NAME
uv run modal run evaluation/deepswe.py::swe_reference_scores --run SPECIALIST_RUN_NAME
```

### T-Rex planner imitation

To train the T-Rex specialist, start from the terminal-agent head. The game supplies three candidate descriptions and marks the planner's preferred option with `Best.`. This experiment trains the action projection to rank those descriptions. Qwen3-8B and the state projection stay frozen. Training uses course seeds 100–111. Development uses 300–303. The checkpoint is chosen by worst-action recall on development decisions, with overall exact agreement and cross-entropy breaking ties. The stock real-time harness treats jumping and running as equivalent while airborne.

```bash
uv run modal run -m evaluation.trex::specialize --parent post-terminal-870f73d383
uv run modal run -m evaluation.trex::measure_agreement --run released-clm --seed 7000
```

The first command saves the new `runs/post-trex-…/best.pt`, evaluates seeds 0–4 with both shield settings, and evaluates seeds 7000–7004 with the shield enabled. Each run records its five individual course results. The second command runs the released CLM on the same fresh courses.

The selected head, `post-trex-980baf6b0c`, recorded 100.0% mean shielded agreement on both sets of courses. On the fresh courses, released CLM recorded 66.9%. Both heads survived all five. On the workshop courses without the shield, the specialist recorded six deaths and zero crash-free courses. Its standard DeepSWE verifier averaged 29.6/38 across five seeds. [Course results, checkpoint hash and costs](evaluation/results/trex_specialization_2026-11-21.json)

### Measured post-training checkpoints, 21 November 2026

Both heads are on `nova-data` under `runs/<name>/best.pt`. The terminal head started at step 546 of this project's mid run. The coding specialist started from the selected terminal head.

| Head | Run name | Tricky Q&A top-1, 191,161 rows | SWE-bench Verified, expected best-of-4 success on 95 issues | DeepSWE, mean resolved out of 38 after standard verifier tuning |
|---|---|---:|---:|---:|
| Terminal-agent post-training | `post-terminal-870f73d383` | 70.75% | 70.67% | 30.2 |
| Coding specialist | `post-swe-ecc0c007a5` | 70.76% | 74.01% | 28.8 |
| CLM | Released post head | 68.63% | 69.42% | 31.0 |

The specialist used labeled SWE-bench training issues. Its development split selected step 50, and the publisher's separate 95-issue split supplied the test result. The CLM control used its released post head on that test. DeepSWE used seeds 1234, 1, 2, 3 and 4 on the same 38 tasks. The local mid head also averaged 30.2 with the same verifier. The experimental outcome adjustment lowered the terminal head's average to 29.4. [Machine-readable results](evaluation/results/post_training_2026-11-21.json)

### Setup

```bash
uv sync
uv run modal setup
```

Run every command below from the repo root, with the virtual environment active (or prefix each one with `uv run`).

### Local or remote

Each folder has a `config.toml` whose `run_mode` decides where its scripts run:

| `run_mode` | Where | Data folder | What it is for |
|---|---|---|---|
| `"local"` (default) | this machine | `downloads/` | small samples, to check that a script works |
| `"remote"` | Modal | the `nova-data` volume, at `/data` | the full run |

Set `run_mode = "remote"` in `data/config.toml`, `finetuning/config.toml` and `evaluation/config.toml` before the full pipeline. Use `--detach` for long runs, so they keep going after you close the terminal. Follow them on the Modal dashboard.

### The full pipeline, in order

Each step reads what the step before it wrote.

| # | Step | Command | Reads | Writes on the volume |
|---|---|---|---|---|
| 1 | Download the Q&A vectors, hard negatives and the authors' heads | `modal run --detach data/load_data.py` | Hugging Face | `qa/`, `neg/`, `heads/` |
| 2 | Pre-training | `modal run --detach -m finetuning.pre_training_handler` | `qa/` | `runs/pre/` |
| 3 | Mid-training | `modal run --detach -m finetuning.mid_training_handler` | `qa/`, `neg/`, `runs/pre/best.pt` | `runs/mid/` |
| 4 | Build agent (state, action) pairs | `modal run --detach data/load_agent_data.py` | ADP v2 on Hugging Face | `agent/pairs/` |
| 5 | Embed the agent pairs with Qwen3-8B | `modal run --detach data/embed_agent.py` | `agent/pairs/` | `qwen3-8b/`, `agent/vec/` |
| 6 | Post-training | `modal run --detach -m finetuning.post_training_handler` | `qa/`, `runs/mid/best.pt`, `agent/vec/` | `runs/post/` |
| 7 | DeepSWE verifier test | `modal run --detach evaluation/deepswe.py::run_deepswe --init post` | `runs/post/best.pt` | `deepswe/post/seed_1234/` |

Steps 4 and 5 don't depend on steps 2 and 3, so they can run at the same time. Step 6 needs all of them.

Each stage directory in `runs/<stage>/` holds:
- `best.pt`: the weights with the best validation score. The next stage starts from it.
- `latest.pt`: weights, optimizer and counters, used to resume.
- `log.jsonl`: one line per evaluation.

## Repo layout

```text
data/         downloads, agent pairs and their embeddings, with config.toml
finetuning/   the three training stages and their shared trainer, with config.toml
evaluation/   DeepSWE verifier test, with config.toml
serving/      using the model: local inference, the decide command, an optional Modal server, with config.toml
schemas/      Pydantic models for every config, batch and result
downloads/    local-mode data (git-ignored)
```

## Credits and licences

CLM is by Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré and Azalia Mirhoseini. Please cite [their work](https://contrastive-lm.notion.site). The Nemotron-derived data is under the NVIDIA Open Data License. The code in this repo is under the Apache 2.0 licence.
