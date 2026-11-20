# System One Model from Scratch

A rebuild of [CLM-8B](https://contrastive-lm.notion.site), an open System One model from Stanford and NVIDIA Research, made for a live workshop by [Dr. Raj Dandekar](https://vizuara.ai) and the team at Vizuara AI Labs.

A System One model does not write text. You give it a situation and a list of options, and it returns a probability for each option in one pass. [Jev](https://typesafe.ai) made the idea popular but is closed source, so this repo builds an open equivalent.

**Site with games and results:** https://system-one-workshop.vercel.app

## The model

```text
Question + document -> Qwen3-8B (frozen) -> State head  (9.44M) -> 512-d
Answer              -> Qwen3-8B (frozen) -> Action head (9.44M) -> 512-d
```

We take the hidden state of the last token (4096 numbers), normalize it, and pass it through a small network: 4096 -> 1536 -> 1536 (LayerNorm) -> 512, with GELU. The score is the cosine of the two vectors times a learned scale. Only the two heads are trained, about 18.9M parameters.

## What was public, and what we rebuilt

The [official repo](https://github.com/Contrastive-LM/CLM) has the serving engine, a fine-tuning script and the T-Rex game. It does not have the pre-training or mid-training code. We recovered those recipes from the settings stored inside the authors' public checkpoints ([jackyk02/qwen3_8b_head](https://huggingface.co/jackyk02/qwen3_8b_head)) and wrote our own trainer.

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

### What each script does

**`data/load_data.py`**: copies the precomputed Q&A vectors and hard negatives from Hugging Face, grouped by chunk, plus the authors' three published heads. Files already downloaded at the right size are skipped. Sources and revisions are in `data/config.toml`.

**`finetuning/pre_training_handler.py`, `mid_training_handler.py`, `post_training_handler.py`**: the three training stages. They share one trainer (`finetuning/training_handler.py`). Each stage's settings are in its section of `finetuning/config.toml`: `[pre]`, `[mid]`, `[post]`.
- **Pre** trains both heads from scratch on every Q&A chunk except the held-out ones.
- **Mid** starts from `runs/pre/best.pt` and adds up to 10 hard negatives per pair. Only `shard_00` has negatives.
- **Post** starts from `runs/mid/best.pt` and mixes agent steps (60%) with Q&A replay (40%). The replay comes from the shard_00 chunks and keeps their hard negatives. It runs 400 steps at lr 0.0003, with the temperature frozen.
- `[pre]` and `[post]` stop at the step counts the original README ran (30,000 and 400), not at their code defaults.
- A crashed or stopped run resumes from `latest.pt` when you run the same command again. To start a stage over, delete its folder: `modal volume rm -r nova-data runs/<stage>`.

**`data/load_agent_data.py`**: streams the [ADP v2](https://huggingface.co/datasets/neulab/adp-v2) agent recordings and keeps up to 30,000 (state, action) text pairs per subset, at most 6 per trajectory. 3% of trajectories are held out for validation. Subsets that are already built are skipped. Settings are in `[agent_pairs]`.

**`data/embed_agent.py`**: embeds every pair with frozen Qwen3-8B on L40S GPUs, the same way the Q&A data was embedded: last token, L2-normalized, float16. States keep their last 2047 tokens and actions their first 2047. It needs vLLM on a GPU, so it only runs with `run_mode = "remote"`. Subsets that are already embedded are skipped. Settings are in `[agent_embedding]`.

**`evaluation/deepswe.py`**: two entrypoints.
- `run_deepswe` fine-tunes a verifier from a head on 22,576 passing DeepSWE steps, using the CLM repo's own `finetune.py`. Then `bon_eval.py` picks the best of 4 rollouts for each of 38 held-out tasks. It always runs on Modal.
  - `--init post` (default) starts from `runs/<name>/best.pt`.
  - `--init authors` starts from the authors' checkpoint. Run this first as a control: it should come near 31/38.
  - `--seed N` changes the seed (default 1234). The score moves by up to 3 tasks between seeds.
  - The result is in `deepswe/<init>/seed_<seed>/result.json`.
- `inspect_deepswe` prints what a verifier checkpoint stores (settings, logit scale, metrics) and the files next to it. With no arguments it shows the authors' verifier, and works locally. `--head post` shows ours and needs `run_mode = "remote"`.

```bash
modal run --detach evaluation/deepswe.py::run_deepswe --init authors
modal run --detach evaluation/deepswe.py::run_deepswe --init post --seed 1235
modal run evaluation/deepswe.py::inspect_deepswe
```

### Quick checks on your machine

With `run_mode = "local"`:

```bash
uv run python -m finetuning.smoke_test                 # pre -> mid -> post -> resume on fake data, a few seconds
modal run data/load_data.py                            # one chunk per source into downloads/
modal run data/load_agent_data.py                      # 200 pairs from each of 2 subsets into downloads/agent/pairs/
modal run evaluation/deepswe.py::inspect_deepswe       # the authors' verifier head
```

The same commands are in `.vscode/launch.json` for the VS Code debugger.

Not ported yet from the original repo: `final_eval.py` (tricky Q&A test) and `run_trex.py` (T-Rex game).

## Results

These are the original repo's numbers ([VizuaraAI/system-one-from-scratch](https://github.com/VizuaraAI/system-one-from-scratch)), measured on 6 November 2026 using 191,161 held-out pairs for the tricky question test.

| Test | Published | Authors' weights, our test | Our model |
|---|---|---|---|
| Tricky Q&A, after pre-training | 52.1% | 51.7% | **51.4%** |
| Tricky Q&A, after mid-training | 69.2% | 69.2% | **68.5%** |
| Tricky Q&A, after post-training | 68.5% | 68.6% | **69.0%** |
| DeepSWE, seed 1234 | 31/38 | 31/38 | **31/38** |
| DeepSWE, mean of 5 seeds | not published | 31.0 | **30.6** |
| T-Rex, median latency | 16.5 ms | 17.1 ms | **17.1 ms** |
| T-Rex, courses survived, shield on | 5/5 | 4/5 | **3/5** |
| T-Rex, planner agreement, shield on | 65.8% | 66.8% | **10%** |
| T-Rex, deaths over 5 courses, shield off | not published | 26 | **50** |

**The known gap:** the Q&A and DeepSWE results replicate, but our model reads the T-Rex labels far worse than the authors' head. The likely cause is post-training, whose full recipe and exact data are not public.
We also tried actions cut down to the bare decision, without the agent's narration (`pipeline/embed_actions.py`). In short 30-second game tests, planner agreement stayed at 16% to 29% with the shield on, against about 80% for the authors' head, so that did not close the gap.

We could not rerun WikiRacing, Super Mario, BFCL or Terminal-Bench, because their harnesses or candidate sets are not public. The Jev numbers we cite come from the authors' logs and charts. We did not run Jev ourselves.

## What we learned

1. **Read the checkpoints.** The settings stored in them reproduced the pre-training and mid-training curve.
2. **Replay needs hard negatives.** With plain Q&A replay, tricky-question accuracy fell to 59.8%. With hard negatives in the replay it held at 69.0%. The blog does not say what its replay contained, so this is our inference.
3. **The README fine-tune misses its own head.** The default dataset, with 405,919 steps, gave 28/38, the same as random. The 22,576 pairs in the head's metadata gave exactly 31/38, and matched the stored validation metric to every digit.
4. **The README's evaluation dataset returns 401.** We used the public copy, `tarsur385/deepswe-prm-embeddings-8k`.
5. **The seed matters.** On 38 tasks, the seed alone moved the score by up to 3 tasks.
6. **The safety shield does a lot of work.** The released head drops from 4/5 courses to 0/5 without it.

## Repo layout

```text
data/         downloads, agent pairs and their embeddings, with config.toml
finetuning/   the three training stages and their shared trainer, with config.toml
evaluation/   DeepSWE verifier test, with config.toml
schemas/      Pydantic models for every config, batch and result
downloads/    local-mode data (git-ignored)
```

## Credits and licences

CLM is by Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré and Azalia Mirhoseini. Please cite [their work](https://contrastive-lm.notion.site). The Nemotron-derived data is under the NVIDIA Open Data License. Vizuara is not affiliated with the authors or with TypeSafe. The code in this repo is under the Apache 2.0 licence.
