# 🎙️ VoicePrint — QLoRA Fine-Tuning

A learning-focused setup for fine-tuning small language models with **LoRA** + **4-bit quantization** (QLoRA).  Start tiny with Qwen3-0.6B, learn the concepts, then scale up.

## End-to-End Workflow

```bash
# 1. Create the environment and install dependencies
uv venv
uv sync

# 2. Add your training data (see data/README.md)
cp data/example.jsonl data/train.jsonl

# 3. Fine-tune the QLoRA adapter
uv run python -m voiceprint.train

# 4. Test the adapter before exporting
uv run python -m voiceprint.eval

# 5. Inspect training metrics (optional)
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

Training saves the small LoRA adapter to `adapters/user-voice-v1`. MLflow
also records the training run, but llama.cpp does not load the MLflow model
directly. For llama.cpp, merge the adapter into the base model and export a
standalone GGUF file.

### Export to GGUF for llama.cpp

Install or build a current [llama.cpp](https://github.com/ggml-org/llama.cpp)
checkout with Qwen3 support. Set `LLAMA_CPP_DIR` to its location, then run:

```bash
export LLAMA_CPP_DIR=/path/to/llama.cpp

# 6. Merge the adapter and print the conversion command
uv run voiceprint-gguf \
	--adapter-dir adapters/user-voice-v1 \
	--merged-dir models/user-voice-merged \
	--output-dir models/user-voice-gguf \
	--outfile user-voice-f16.gguf \
	--outtype f16 \
	--llama-cpp-dir "$LLAMA_CPP_DIR"

# 7. Convert the merged Hugging Face model to an intermediate GGUF
uv run python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
	models/user-voice-merged \
	--outfile models/user-voice-gguf/user-voice-f16.gguf \
	--outtype f16

# 8. Quantize the intermediate GGUF for a smaller llama.cpp model
"$LLAMA_CPP_DIR/llama-quantize" \
	models/user-voice-gguf/user-voice-f16.gguf \
	models/user-voice-gguf/user-voice-q4_k_m.gguf \
	Q4_K_M

# 9. Run the voice model
"$LLAMA_CPP_DIR/llama-cli" \
	-m models/user-voice-gguf/user-voice-q4_k_m.gguf \
	-p "Hey, what are you working on?" \
	-n 256 \
	-t 8
```

The conversion is intentionally two-stage: `convert_hf_to_gguf.py` creates a
full-precision GGUF, and `llama-quantize` creates the final `Q4_K_M` file.
The merged model directory must include the tokenizer files; the
`voiceprint-gguf` command copies them from the adapter automatically.

If llama.cpp reports `BPE pre-tokenizer was not recognized`, update the
llama.cpp checkout. This usually means the converter is older than the Qwen3
tokenizer support or the merged directory is missing its tokenizer files.

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

1. Curate your voice dataset in `data/`.
2. Train and test the adapter.
3. Merge and convert it to GGUF.
4. Run the quantized model with llama.cpp.
