# Limitless Autonomous Guardian (LAG)

LAG is a ground-up, solo-friendly research sandbox for building a foundation language model without relying on pre-existing checkpoints. The initial prototype focuses on:

- ⚙️ **Tokenizer** – pure Python byte-pair encoder you can train on any text corpus.
- 🧠 **Decoder-only transformer** – compact GPT-style architecture implemented from scratch in PyTorch.
- 📚 **Data pipeline** – utilities to fetch the Tiny Shakespeare corpus and turn it into autoregressive training batches.
- 🧪 **Training harness** – CLI entrypoint that stitches everything together so you can iterate locally on CPU or tap into free GPU tiers.

This repository is intentionally lightweight so you can extend it toward the full Limitless Autonomous Guardian vision as resources grow.

---

## Quickstart

> Tested with Python 3.9+ on Ubuntu 24.04 inside a VS Code dev container.

### 1. Install dependencies

```bash
pip install -e .[dev]
```

### 2. Train the tokenizer and model on Tiny Shakespeare (baseline)

```bash
python -m lag.train --max-steps 50 --batch-size 8 --seq-len 128 --device cpu
```

The script will:

1. Download the corpus into `data/` (if missing).
2. Fit a BPE tokenizer (default vocab size 2,048).
3. Tokenize the corpus and build train/validation loaders.
4. Train the decoder-only transformer and periodically report losses.
5. Save artifacts (tokenizer + model weights) under `artifacts/`.

### 3. Sample from the trained model

```bash
python -m lag.infer "Guardians of tomorrow" --max-new-tokens 80 --device cpu
```

The CLI loads the latest artifacts in `artifacts/` and prints a continuation of your prompt so you can sanity-check behavior quickly.

### 4. Run tests

```bash
pytest
```

---

## Project structure

- `lag/tokenizer.py` – BPE tokenizer implementation with save/load helpers.
- `lag/data.py` – dataset downloader, HuggingFace streaming helpers, and `NextTokenDataset` for next-token prediction.
- `lag/model.py` – GPT-style transformer blocks with rotary attention, RMSNorm, and configurable activations.
- `lag/train.py` – configuration-driven training loop with gradient accumulation, LR schedules, and dataset mixing.
- `tests/` – smoke tests for tokenizer, data pipeline, and model forward pass.

---

## Scaling to a larger LAG model

- **Tokenizer** – Increase `--vocab-size`, enable `--lowercase`, and mix domain corpora through `--hf-source`.
- **Architecture** – Tune `--d-model`, `--n-layers`, `--n-kv-heads`, activation (`--activation`), and rotary settings to match target parameter counts.
- **Optimization** – Adjust `--micro-batch-size`, `--grad-accum`, and `--scheduler` to fit within free GPU quotas.
- **Safety & Alignment** – Add chat and safety data sources with dedicated instruction/response fields, then layer alignment techniques.


### Suggested open datasets

| Domain | Dataset | Notes |
| --- | --- | --- |
| Dialogue | `HuggingFaceH4/ultrachat_200k` | High-quality assistant conversations, instruction/response fields.
| Safety | `Anthropic/hh-rlhf` | Harmless/helpful preference data; filter for desired policies.
| Science | `allenai/science-qa` | QA pairs across STEM topics, multi-choice reasoning.
| General knowledge | `openai/gpt2` or `fineweb-edu` | Clean web text for world knowledge.
| Human basics | `bigscience/P3` subsets (e.g., social IQa) | Social reasoning prompts.

Add sources via repeated `--hf-source` flags, for example:

```bash
python -m lag.train \
		--hf-source dataset=HuggingFaceH4/ultrachat_200k,split=train_sft,instruction_field=instruction,response_field=response \
	--hf-source dataset=allenai/science-qa,text_field=text \
	--hf-source dataset=bigscience/P3,subset=super_glue_copa,text_field=input,weight=0.5 \
	--seq-len 2048 --d-model 1024 --n-heads 16 --n-layers 24 --gradient-checkpointing \
	--micro-batch-size 2 --grad-accum 32 --device cuda
```

---

---

## Roadmap

| Phase | Goal | Notes |
| --- | --- | --- |
| Prototype | Stand up local training + evaluation | ✅ covered in this repo |
| Scaling | Acquire sustained GPU access | Apply for TPU/GPU research credits and optimize gradients |
| Alignment | Human / synthetic feedback loops | Design safety rubrics and reward models |
| Deployment | Guarded API + monitoring | Add policy enforcement, logging, and fallback heuristics |

Contributions and experiment results are welcome—document findings and share back to strengthen the autonomous guardian.