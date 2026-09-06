#!/usr/bin/env python3
"""Main training script — QLoRA fine-tuning pipeline.

ELI5 — The full training pipeline step by step:
───────────────────────────────────────────────

1. Load the base model at 4-bit precision (this is the "quantization" part).
   The model lives on your GPU in a compressed form.

2. Attach LoRA adapters to specific layers (this is the "LoRA" part).
   Only the LoRA matrices are trainable — the base model stays frozen.

3. Load and tokenize your data (the voice dataset — examples of your writing).

4. Train: the optimizer updates only the LoRA weights to make the model
   generate text that sounds more like you.

5. Save: only the tiny LoRA adapter is saved (usually < 50 MB).
   Later you merge it into the base model or load it alongside for inference.

Key references:
    QLoRA paper:   https://arxiv.org/abs/2305.14314
    PEFT docs:     https://huggingface.co/docs/peft/developer_guides/quantization
    bitsandbytes:  https://huggingface.co/docs/bitsandbytes
    MLFlow:        https://mlflow.org/docs/latest/quickstart.html
"""

from __future__ import annotations

import logging
from pathlib import Path

# Load .env if present (HF_TOKEN, MLFLOW_DISABLE_AGENT_HINT, etc.).
from dotenv import load_dotenv
load_dotenv()

import mlflow
import mlflow.pytorch
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.panel import Panel
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from voiceprint.config import (
    BASE_MODEL_ID,
    DataConfig,
    LoraConfig as LoraCfg,
    MlflowConfig,
    TrainingConfig,
    VOICE_NAME,
)
from voiceprint.data import load_and_prepare

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("voiceprint")

# Grab configs (all defaults, override programmatically or via env later).
lora_cfg = LoraCfg()
train_cfg = TrainingConfig()
data_cfg = DataConfig()
mlflow_cfg = MlflowConfig()


def load_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Step 1 — Load the base model quantized to 4-bit.

    ELI5: We're loading Qwen3-0.6B in a compressed format (NF4 = 4-bit).
    The bitsandbytes library does the heavy lifting — it reads the 16-bit
    weights from disk and stores them as 4-bit values on the GPU.  The
    quality loss is small because transformers have redundant representations.

    VRAM usage for Qwen3-0.6B at 4-bit:
        Model weights:      ~2 GB
        Activations (batch): ~1-2 GB (depends on batch size + seq length)
        Optimizer states:    ~0.5 GB (paged to CPU with paged_adamw_8bit)
        Total:               ~5-8 GB out of 32 GB (plenty of headroom!)
    """
    console.print(Panel.fit(
        "[bold cyan]Step 1[/bold cyan]: Loading base model (4-bit quantized)…",
        subtitle=f"Voice: {VOICE_NAME} | Model: {BASE_MODEL_ID}",
    ))

    # Quantization config — tells transformers how to compress the model.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"  Model loaded: [green]{model.config.model_type}[/green] ({model.config.tie_word_embeddings})")
    console.print(f"  Trainable params before LoRA: [red]0 (all frozen at 4-bit)[/red]")

    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
) -> AutoModelForCausalLM:
    """Step 2 — Attach LoRA adapters to the quantized model.

    ELI5: Imagine the model's weight matrix W as a wall painting.
    LoRA doesn't repaint the wall — it holds up a transparency (two
    small matrices A and B) in front of it.  The effective transformation
    becomes:  W_eff = W + B @ A

    During training only A and B are updated.  W stays frozen at 4-bit.
    At inference you either:
      a) Keep the LoRA adapter loaded alongside the base model, or
      b) Merge B @ A into W permanently (loses the ability to swap adapters).

    prepare_model_for_kbit_training() is a safety step: it casts certain
    layers (like LayerNorm) to higher precision so gradients don't
    explode when the base weights are only 4-bit.
    """
    console.print(Panel.fit(
        "[bold cyan]Step 2[/bold cyan]: Attaching LoRA adapters…",
        subtitle=f"r={lora_cfg.r}, alpha={lora_cfg.alpha}",
    ))

    # Safety prep for 4-bit training — casts norm layers to FP32.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=train_cfg.gradient_checkpointing,
    )

    # Create the LoRA config and attach it.
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type="CAUSAL_LM",  # We're doing next-token prediction.
    )

    model = get_peft_model(model, peft_config)

    # Show trainable vs frozen params.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"  Total params:      {total:,}")
    console.print(f"  Trainable (LoRA):  [green]{trainable:,}[/green] ({100 * trainable / total:.2f}%)")
    console.print(f"  Frozen (base):     {total - trainable:,}")

    model.print_trainable_parameters()  # PEFT's own summary table.

    return model


def setup_mlflow() -> None:
    """Configure MLFlow tracking.

    ELI5: MLFlow records every experiment run so you can compare
    different settings (learning rate, rank, data) later.  Start the
    UI with:  uv run mlflow ui --backend-store-uri mlflow/
    """
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)
    mlflow.start_run(run_name=f"qlora-{VOICE_NAME}-{BASE_MODEL_ID.split('/')[-1]}-r{lora_cfg.r}")

    # Log all training hyperparameters.
    mlflow.log_params({
        "base_model": BASE_MODEL_ID,
        "lora_r": lora_cfg.r,
        "lora_alpha": lora_cfg.alpha,
        "lora_dropout": lora_cfg.dropout,
        "learning_rate": train_cfg.learning_rate,
        "max_steps": train_cfg.max_steps,
        "batch_size": train_cfg.per_device_train_batch_size,
        "gradient_accumulation": train_cfg.gradient_accumulation_steps,
        "max_seq_length": train_cfg.max_seq_length,
    })
    console.print("[dim]MLFlow run started — check mlflow/ for artifacts.[/dim]")


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> None:
    """Steps 3-5 — Load data, train, save.

    ELI5: We use HuggingFace's SFTTrainer (Supervised Fine-Tuning Trainer).
    It's a wrapper around the standard Trainer that handles the common
    patterns for instruction-following / causal LM training.

    The training loop:
      1. Sample a batch of tokenized texts.
      2. Forward pass: model predicts each token given previous tokens.
      3. Loss: cross-entropy between predicted and actual tokens.
      4. Backward pass: gradients flow through LoRA matrices only.
      5. Optimizer step: update LoRA weights.
      6. Repeat until max_steps.
    """
    console.print(Panel.fit(
        "[bold cyan]Step 3[/bold cyan]: Loading training data…",
        subtitle=str(data_cfg.train_file),
    ))

    dataset = load_and_prepare(data_cfg)

    console.print(Panel.fit(
        "[bold cyan]Step 4[/bold cyan]: Training (QLoRA)…",
        subtitle=f"max_steps={train_cfg.max_steps}, lr={train_cfg.learning_rate}",
    ))

    # SFTConfig is the current config object for TRL 1.x.  It keeps the
    # same training knobs, but the trainer constructor now expects the config
    # passed via `args` and the tokenizer via `processing_class`.
    sft_config = SFTConfig(
        output_dir=str(train_cfg.output_dir),
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_steps=int(train_cfg.max_steps * train_cfg.warmup_ratio),
        max_steps=train_cfg.max_steps,
        num_train_epochs=train_cfg.num_train_epochs,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        optim=train_cfg.optim,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        seed=train_cfg.seed,
        bf16=False,
        fp16=False,
        report_to="mlflow",
        disable_tqdm=False,
        dataset_text_field="text",
        max_length=train_cfg.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Step 5 — Save the LoRA adapter.
    console.print(Panel.fit(
        "[bold cyan]Step 5[/bold cyan]: Saving LoRA adapter…",
        subtitle=str(train_cfg.output_dir),
    ))

    # Save in PEFT format (adapter only — tiny!).
    model.save_pretrained(str(train_cfg.output_dir))
    tokenizer.save_pretrained(str(train_cfg.output_dir))

    console.print(f"\n[bold green]✓ Done![/bold green] Adapter saved to {train_cfg.output_dir}")

    # Save the trained model object itself.  This is the MLflow PyTorch flow from
    # the official docs: the registry can later reload it with
    # mlflow.pytorch.load_model(f"runs:/{run_id}/pytorch_model").
    mlflow.pytorch.log_model(
        model,
        name="pytorch_model",
        serialization_format="pickle",
    )

    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    console.print(f"  [dim]MLFlow run ID: {run_id}[/dim]")
    console.print(f"  [dim]Load with:     mlflow.pytorch.load_model('runs:/{run_id}/pytorch_model')[/dim]")

    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"  [dim]  View run:  uv run mlflow ui --backend-store-uri {mlflow_cfg.tracking_uri}[/dim]")
    console.print(f"  [dim]  Test:      python -c 'from voiceprint.eval import main; main()'[/dim]")
    console.print(f"  [dim]  Merge:     see evals/README.md for merging adapter into base model[/dim]")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full QLoRA training pipeline."""
    console.print(Panel(
        "[bold yellow]voiceprint[/bold yellow] — QLoRA Fine-Tuning\n"
        f"Base model: {BASE_MODEL_ID}\n"
        f"GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
        f"VRAM:       {torch.cuda.get_device_properties(0).total_memory // 1024**2 if torch.cuda.is_available() else 0} MB",
        title="🎙️ VoicePrint QLoRA Trainer",
    ))

    if not torch.cuda.is_available():
        console.print("[bold red]Error: No NVIDIA GPU detected. QLoRA requires a CUDA GPU.[/bold red]")
        raise SystemExit(1)

    # Create output directories.
    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    mlflow_cfg.artifact_location.mkdir(parents=True, exist_ok=True)

    setup_mlflow()
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    train(model, tokenizer)


if __name__ == "__main__":
    main()
