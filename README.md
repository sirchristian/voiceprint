# 🎙️ VoicePrint — QLoRA Fine-Tuning

A learning-focused setup for fine-tuning small language models with **LoRA** + **4-bit quantization** (QLoRA).  Start tiny with Qwen3-0.6B, learn the concepts, then scale up.

## Quick Start

```bash
# 1. Create the virtual environment (uv handles this)
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv sync

# 3. Add your training data (see data/README.md)
cp data/example.jsonl data/train.jsonl

# 4. Train
uv run python -m voiceprint.train

# 5. Test the adapter
uv run python -m voiceprint.eval

# 6. View training metrics
mlflow ui --backend-store-uri mlflow/
```

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `src/voiceprint/` | Core Python package (train, eval, config, data) |
| `data/` | Your training JSONL files (see [data/README.md](data/README.md)) |
| `adapters/` | Saved LoRA adapter checkpoints (tiny — usually < 50 MB each) |
| `models/` | → symlink to `/mnt/storage/models/voiceprint/` (base models + HF cache) |
| `evals/` | Evaluation scripts, GGUF conversion, llama.cpp testing |
| `mlflow/` | MLFlow experiment tracking (runs, metrics, artifacts) |

## ELI5 — What is QLoRA?

### LoRA (Low-Rank Adaptation)
Instead of fine-tuning *all* parameters in a model (which requires saving a full copy), LoRA adds **tiny trainable matrices** alongside the frozen base weights.  Think of it as a sticker overlay on the original model — you can swap stickers without buying a new model.

### Quantization (4-bit)
Compresses the base model weights from 16-bit to 4-bit floating point.  The model takes ~4× less VRAM with minimal quality loss.  NF4 (NormalFloat 4) is the smartest 4-bit format — it assigns more precision to the weight values that matter most.

### QLoRA = LoRA + 4-bit Quantization
Train LoRA adapters on a quantized base model.  The result: you can fine-tune a 70B model on a single consumer GPU.

### Key Resources
- **LoRA paper**: https://arxiv.org/abs/2106.09685
- **QLoRA paper**: https://arxiv.org/abs/2305.14314
- **PEFT docs (quantization guide)**: https://huggingface.co/docs/peft/developer_guides/quantization
- **bitsandbytes docs**: https://huggingface.co/docs/bitsandbytes

## Hardware

Built for **NVIDIA RTX 5090 (32 GB VRAM)**.  The 32 GB gives plenty of headroom:

| Model | 4-bit VRAM | 8-bit VRAM | 16-bit VRAM |
|-------|-----------|-----------|------------|
| Qwen3-0.6B | ~2 GB | ~4 GB | ~1.2 GB |
| Qwen2.5-1.5B | ~4 GB | ~8 GB | ~3 GB |
| Qwen2.5-7B | ~18 GB | ~36 GB | ~14 GB |

## Configuring

All settings live in `src/voiceprint/config.py` — every parameter has an ELI5 docstring explaining what it does and how to tune it.  Key knobs:

- **`LoraConfig.r`** — LoRA rank (higher = more trainable params, more capacity)
- **`TrainingConfig.learning_rate`** — Start at 2e-4 for LoRA
- **`TrainingConfig.max_steps`** — Training duration
- **`TrainingConfig.max_seq_length`** — Max tokens per example (affects VRAM)

## Next Steps

1. Curate your Chris dataset → `data/`
2. Run a quick training run → `adapters/`
3. Test the adapter → `eval/`
4. Merge + convert to GGUF for llama.cpp → `evals/README.md`
