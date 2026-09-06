# AGENTS.md

## Purpose

VoicePrint is a QLoRA fine-tuning pipeline for training a "personal voice" model — a small language model fine-tuned on someone's writing to capture their style and tone.  The end goal is a GGUF file runnable in llama.cpp.

## Tech Stack

- **Python 3.12+** managed by `uv`
- **PyTorch** + **transformers** (HuggingFace) for model loading
- **PEFT** + **bitsandbytes** for LoRA + 4-bit quantization (QLoRA)
- **TRL** for the SFTTrainer
- **MLFlow** for experiment tracking (SQLite backend in `mlflow/`)
- **jaxtyping** for runtime tensor shape type hints

## Directory Layout

| Path | Purpose |
|------|---------|
| `src/voiceprint/` | Core package — train, eval, config, data |
| `data/` | Training JSONL files (`train.jsonl` is user-created, gitignored) |
| `adapters/` | Saved LoRA adapter checkpoints (< 50 MB each) |
| `models/` | Symlink → `/mnt/storage/models/voiceprint/` (HF cache + merged models) |
| `evals/` | GGUF conversion guide, llama.cpp testing |
| `mlflow/` | MLFlow SQLite DB + artifacts |
| `training/` | Training artifacts, VRAM guide |

## Key Conventions

- **ELI5 comments**: Every file and config parameter has beginner-friendly explanations.  Preserve this style when editing.
- **Config-driven**: All tunables live in `src/voiceprint/config.py`.  `VOICE_NAME` controls output paths and run names — make it generic, never hardcode a person's name.
- **Open source**: This is a public MIT-licensed project.  No Campusesp references, no hardcoded names.
- **Agent commits**: Sign with `Co-Authored-By: pi/<model> <agent@techbychris.com>`.

## Gotchas

- `SFTTrainer` is imported from `trl`, not `transformers`.
- `transformers` v5 dropped `warmup_ratio` — use `warmup_steps` (computed from the ratio in config).
- `SFTTrainer` handles tokenization internally via `dataset_text_field` — don't pre-tokenize in `data.py`.
- The `models/` directory is a symlink to `/mnt/storage/models/voiceprint/` — don't break it.
- MLFlow tracking URI is `sqlite:///mlflow/mlflow.db` — use `uv run mlflow ui` to start the dashboard.

## Running

```bash
uv venv && source .venv/bin/activate
uv sync
cp data/example.jsonl data/train.jsonl   # or add real data
uv run python -m voiceprint.train         # train
uv run python -m voiceprint.eval          # test adapter
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db  # metrics
```
