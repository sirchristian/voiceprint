"""Configuration for QLoRA training.

ELI5 — What are we tuning and why?
──────────────────────────────────

Think of a language model as a giant book of "knowledge about how words go
together."  Fine-tuning means writing a few extra pages at the back that
teach the model something new.  LoRA (Low-Rank Adaptation) is the clever
part: instead of rewriting the whole book, LoRA adds a thin overlay of
*small* matrices that sit on top of the original weights.  The original
weights stay frozen.  This makes fine-tuning:
    1. Cheap — only ~0.1-1% of parameters are trainable.
    2. Fast — the overlay matrices are tiny.
    3. Swappable — you can load/unload overlays without touching the base model.

Quantization compresses the (frozen) base model from 16-bit to 4-bit
floating point.  The numbers take less memory and still produce nearly
identical outputs.  Combined with LoRA this is called QLoRA and lets us
fit much bigger models on a single GPU.

Key papers / docs:
    LoRA:  https://arxiv.org/abs/2106.09685
    QLoRA: https://arxiv.org/abs/2305.14314
    PEFT:  https://huggingface.co/docs/peft/developer_guides/quantization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ──────────────────────────────────────────────────────────────────────
# Identity
# ──────────────────────────────────────────────────────────────────────
# What to call this voice profile.  Used for adapter directory names,
# MLFlow run names, and log messages.  Change this per person/project.
VOICE_NAME: str = "user"

# ──────────────────────────────────────────────────────────────────────
# Base model
# ──────────────────────────────────────────────────────────────────────
# Qwen3.5-9B: hybrid Gated-DeltaNet/attention vision-language model.
# `AutoModelForCausalLM` loads only its text backbone (no vision tower),
# which is what we want for a text-only LoRA — cheaper and simpler.
# At 4-bit quantization that's ~5-6 GB weights, ~10-14 GB loaded on GPU.
BASE_MODEL_ID: str = "Qwen/Qwen3.5-9B"

# Where the base model is cached (symlinked to the big spinning drive).
MODELS_DIR: Path = Path("models").resolve()


# ──────────────────────────────────────────────────────────────────────
# LoRA hyperparameters
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LoraConfig:
    """LoRA (Low-Rank Adaptation) settings.

    ELI5 — r and alpha:
    ────────────────────
    r (rank):  How many "shortcut dimensions" we add.  Think of the
    original weight matrix as a highway.  LoRA adds a detour with `r`
    lanes.  Small `r` = fewer lanes = less capacity but fewer params.
    Common values: 8, 16, 32, 64.  For a tiny model start with 16.

    alpha:     A scaling factor applied to the LoRA output.  The
    effective strength is `alpha / r`.  PEFT defaults to `alpha == r`,
    giving a 1:1 scale.  You can bump `alpha` without increasing `r`
    to get more expressiveness without more parameters.

    target_modules:  Which layers inside the transformer get LoRA
    overlays.  Every attention layer has several sub-matrices (q, k, v,
    o, gate, up, down).  Targeting all of them gives the most flexibility.

    lora_dropout:  A regularizer.  Randomly zeroes a fraction of LoRA
    activations during training so the model doesn't memorize.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Target every attention + feedforward sub-layer for maximum expressiveness.
    # Qwen3.5 is a hybrid model: 3 in 4 layers use Gated DeltaNet (linear
    # attention, named in_proj_*/out_proj) instead of regular attention
    # (q_proj/k_proj/v_proj/o_proj) — target both so LoRA reaches every layer.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: Literal["none"] = "none"
    # "A" is the modern default — slightly better quality for inference.
    # "A" = NFO4, "A_sq" = NF4-sq.  NF4 is the 4-bit format used in QLoRA.
    # See: https://arxiv.org/abs/2305.14314  (Section 3.1)
    use_dora: bool = False  # DoRA variant (direction-only LoRA) — experimental


# ──────────────────────────────────────────────────────────────────────
# Quantization settings
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BitsAndBytesConfig:
    """4-bit quantization settings via the bitsandbytes library.

    ELI5 — NF4 vs INT4:
    ────────────────────
    INT4 simply truncates weights to 4-bit integers.  NF4 (NormalFloat 4)
    is smarter: it assigns more bits to weight values that appear more
    frequently in transformer distributions (which are roughly normal).
    NF4 consistently gives better quality at the same 4-bit budget.

    llm_int8_skip_modules:  Layers we DON'T quantize because they're
    sensitive to precision loss (typically the final LM head and the
    embedding layer).  Leaving them at full precision avoids quality hits.
    """

    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4"] = "nf4"
    # Use double-quantization: quantize the *quantization constants* too.
    # Saves ~0.5-1 GB more memory.  Quality hit is negligible.
    bnb_4bit_use_double_quant: bool = True
    # Compute in float16 even though weights are 4-bit.  This is the
    # standard setup for NVIDIA GPUs (CUDA cores love FP16).
    bnb_4bit_compute_dtype: str = "float16"


# ──────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainingConfig:
    """Training settings.

    ELI5 — Why these values?
    ────────────────────────
    With LoRA we only train the small overlay, so we can use a higher
    learning rate than full fine-tuning without destabilizing the model.
    The learning rate schedule warms up, then decays so we converge
    smoothly.

    max_steps:     Total optimization steps.  Each step processes one
    batch.  For a tiny model + small dataset this is fine.  Scale up
    for bigger data.

    per_device_train_batch_size:  How many examples per GPU per step.
    With gradient accumulation the *effective* batch size is
    `batch_size × accumulation_steps`.  We use accumulation to get a
    larger effective batch without blowing up VRAM.

    gradient_checkpointing:  Trades compute for memory.  Instead of
    storing all intermediate activations for backprop, it recomputes
    them on the fly.  ~30% slower but ~40% less memory.
    """

    # ── Output ──
    output_dir: Path = Path("adapters/user-voice-v1")

    # ── Optimizer ──
    learning_rate: float = 2e-4          # LoRA-friendly LR (higher than full FT)
    lr_scheduler_type: Literal["cosine"] = "cosine"  # Smooth warmup + decay
    warmup_ratio: float = 0.05           # First 5% of steps warm up

    # ── Batch size ──
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 16

    # ── Duration ──
    max_steps: int = 500                 # ~500 steps for a quick test run
    # Set to 0 to train for a fixed number of epochs instead:
    num_train_epochs: int = 0

    # ── Memory ──
    gradient_checkpointing: bool = True
    optim: Literal["adamw_torch", "paged_adamw_8bit"] = "adamw_torch"
    # The standard AdamW torch optimizer is the simplest stable choice for
    # this tiny QLoRA workflow.  It avoids the 8-bit paged optimizer edge
    # cases that can trip mixed-precision training on some GPU stacks.

    # ── Logging / saving ──
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3  # Keep only the last 3 checkpoints

    # ── Misc ──
    seed: int = 42
    max_seq_length: int = 512  # Token limit per example


# ──────────────────────────────────────────────────────────────────────
# MLFlow tracking
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MlflowConfig:
    """MLFlow experiment tracking settings.

    ELI5 — Why MLFlow?
    ───────────────────
    When you tweak learning rates, ranks, or data, it's easy to forget
    what you tried.  MLFlow logs every run with its params, metrics,
    and artifacts so you can compare them side-by-side in a local
    dashboard: `uv run mlflow ui --backend-store-uri mlflow/`
    """

    tracking_uri: str = "sqlite:///mlflow/mlflow.db"
    artifact_location: Path = Path("mlflow/artifacts")
    experiment_name: str = "voiceprint-qlora"


# ──────────────────────────────────────────────────────────────────────
# Dataset paths
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataConfig:
    """Paths and settings for the training dataset.

    Expected format: a JSONL file where each line is:
        {
          "messages": [
            {"role": "user", "content": "Write a quick email..."},
            {"role": "assistant", "content": "Sure, here's a draft..."}
          ]
        }

    The model's chat template is applied automatically during loading,
    so the dataset is model-agnostic — swap the base model without
    redesigning the data.

    See data/README.md for curation tips.
    """

    train_file: Path = Path("data/train.jsonl")
    val_file: Path | None = None  # Optional validation split
