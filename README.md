# VoicePrint — QLoRA Fine-Tuning

Fine-tune a small language model on your writing to capture your voice.  Uses **LoRA** + **4-bit quantization** so you can do this on a single consumer GPU.

Start with Qwen3-0.6B (~600M params), learn the workflow, then swap in a bigger model when you're ready.

## Quick Start

```bash
# Set up
uv venv && source .venv/bin/activate
uv sync

# Copy example data (or write your own — see data/README.md)
cp data/example.jsonl data/train.jsonl

# Train the LoRA adapter (~5-10 min on a 5090)
uv run voiceprint-train

# Test it — compare base model vs your adapter
uv run voiceprint-eval --base-only   # vanilla model
uv run voiceprint-eval               # model + your voice adapter
```

Training saves a tiny LoRA adapter to `adapters/user-voice-v1/`.  You can load
it alongside the base model for inference, or merge it into a standalone model
for llama.cpp.

### MLFlow (optional)

Track metrics across runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

## Export to GGUF for llama.cpp

`voiceprint-gguf` merges the adapter into the base model and prints the command
to run llama.cpp's converter:

```bash
uv run voiceprint-gguf
```

Output:

```
Merged model saved to: models/user-voice-merged
Run this to convert to GGUF:
uv run python /path/to/llama.cpp/convert_hf_to_gguf.py models/user-voice-merged --outfile ...
```

Copy that command, run it, then quantize if you want a smaller file:

```bash
./llama.cpp/llama-quantize \
  models/user-voice-gguf/user-voice-f16.gguf \
  models/user-voice-gguf/user-voice-q4_k_m.gguf \
  Q4_K_M
```

If llama.cpp complains about `BPE pre-tokenizer was not recognized`, your
llama.cpp checkout is too old for Qwen3. Update it.

## Data Format

One JSONL file, one example per line:

```jsonl
{"messages": [
  {"role": "user", "content": "Write a quick email asking whether there is a recording."},
  {"role": "assistant", "content": "Hey team — I missed standup today..."}
]}
```

The model's chat template is applied automatically — no Qwen-specific tokens
in your data. See [data/README.md](data/README.md) for curation tips.

## Config

Everything tunable lives in `src/voiceprint/config.py`.  Key knobs:

| Setting | What it does |
|---------|-------------|
| `VOICE_NAME` | Label for this voice profile (affects output paths, run names) |
| `BASE_MODEL_ID` | Which HuggingFace model to fine-tune |
| `LoraConfig.r` | LoRA rank — more = bigger adapter, more capacity |
| `TrainingConfig.learning_rate` | Start at 2e-4 for LoRA |
| `TrainingConfig.max_steps` | How long to train |
| `TrainingConfig.max_seq_length` | Token limit per example (affects VRAM) |

Every parameter has an ELI5 docstring.  Read the file — it's designed to be a reference.

## Directory Layout

| Path | Purpose |
|------|---------|
| `src/voiceprint/` | Core package (train, eval, config, data loading) |
| `data/` | Training JSONL (`train.jsonl` is gitignored — add your own) |
| `adapters/` | Saved LoRA checkpoints (< 50 MB each) |
| `models/` | → symlink to `/mnt/storage/models/voiceprint/` (HF cache + merged models) |
| `mlflow/` | MLFlow SQLite DB + artifacts |

## Hardware

Tested on **RTX 5090 (32 GB VRAM)**.  Qwen3-0.6B at 4-bit needs ~5-8 GB total
(weights + activations + optimizer).  Plenty of headroom for a 7B model too.

## Resources

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)

## TODO

- [x] Thinking is preserved automatically — training data uses `enable_thinking=False`
      (strips `<think>` blocks from training examples only), so the LoRA shapes the
      final response style without touching the base model's reasoning behavior.
      Confirmed working well on Qwen3.5-9B via llama.cpp.
