#!/usr/bin/env python3
"""Evaluation / inference script — test your LoRA adapter.

ELI5 — How do we use the trained adapter?
─────────────────────────────────────────

After training we have a small adapter file (usually < 50 MB).  To use
it we:
  1. Load the base model (can be full precision or quantized).
  2. Load the adapter on top with PEFT.
  3. Generate text — the model now has your "voice" baked in.

You can also load the model directly from MLFlow:
    import mlflow.pytorch
    model = mlflow.pytorch.load_model("runs:/<run_id>/model")

Later you'll want to:
  - Merge the adapter into the base model (saves one load step).
  - Convert to GGUF format for llama.cpp (see evals/README.md).
  - Test with `llama-cli` or `llama-server`.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from voiceprint.config import BASE_MODEL_ID, MlflowConfig, TrainingConfig

console = Console()
train_cfg = TrainingConfig()
mlflow_cfg = MlflowConfig()

# Path to the saved adapter (output of train.py).
ADAPTER_DIR = train_cfg.output_dir


def load_from_mlflow(run_id: str | None = None) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load the model directly from MLFlow.

    ELI5: MLFlow stores the model with metadata (flavor, params, signature)
    so you can load it back without knowing the exact file paths or config.
    The run_id tells MLFlow which training run to pull the model from.

    If run_id is None, it loads the latest run from the experiment.
    """
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)

    if run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        # Find the latest run.
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(mlflow_cfg.experiment_name)
        if experiment is None:
            console.print("[bold red]No MLFlow experiment found. Run training first.[/bold red]")
            raise SystemExit(1)
        runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"])
        if not runs:
            console.print("[bold red]No runs found in experiment.[/bold red]")
            raise SystemExit(1)
        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"

    console.print(f"[dim]Loading from MLFlow run: {run_id}[/dim]")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    model.to("cuda")

    # Tokenizer lives in the adapter dir (saved alongside the adapter).
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR),
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]Model loaded from MLFlow[/green]")
    return model, tokenizer


def load_merged_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model + LoRA adapter.

    ELI5: We load the model at 4-bit (same as training) to minimize VRAM.
    For merging into a standalone model you'd load at full precision, but
    that needs ~1.2 GB just for the base — too much when the coding agent
    is also running.  4-bit inference is plenty fast and accurate for
    a tiny model like Qwen3-0.6B.

    To merge into a standalone model (for llama.cpp conversion), load
    without quantization:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, dtype=torch.float16, device_map="auto",
        )
    """
    console.print(Panel.fit(
        "[bold cyan]Loading model + adapter[/bold cyan]",
        subtitle=f"Base: {BASE_MODEL_ID} | Adapter: {ADAPTER_DIR}",
    ))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
    )

    # Stack the LoRA adapter on top (don't merge — keeps model quantized).
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR),
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]Model + adapter loaded (4-bit)[/green]")
    return model, tokenizer


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate text from a prompt.

    ELI5: The model predicts one token at a time.  "Temperature" controls
    randomness: 0.0 = most likely token every time (deterministic),
    higher = more creative but potentially nonsensical.  For a "voice"
    model you might want 0.7-0.9 to sound natural.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():  # No gradients needed for inference.
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,             # Nucleus sampling — ignore unlikely tokens.
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated tokens back to text.
    generated = outputs[0][inputs["input_ids"].shape[1]:]  # Skip the prompt tokens.
    return tokenizer.decode(generated, skip_special_tokens=True)


def quick_test() -> None:
    """Run a few sample prompts to see if the model sounds right."""
    model, tokenizer = load_merged_model()

    prompts = [
        "Hey, I just wanted to check in about",
        "So here's the thing, I think we should",
        "Alright, let me break this down for you —",
    ]

    for prompt in prompts:
        console.print(f"\n[dim]Prompt:[/dim] {prompt}")
        response = generate(model, tokenizer, prompt)
        console.print(f"[green]Response:[/green] {response}")


def main() -> None:
    """Entry point for evaluation."""
    if not ADAPTER_DIR.exists():
        console.print(f"[bold red]Adapter not found at {ADAPTER_DIR}[/bold red]")
        console.print("[dim]Run training first: uv run python -m voiceprint.train[/dim]")
        raise SystemExit(1)

    console.print(Panel(
        "[bold yellow]voiceprint[/bold yellow] — Eval / Inference\n"
        f"Adapter: {ADAPTER_DIR}",
        title="🎙️ VoicePrint Eval",
    ))

    quick_test()


if __name__ == "__main__":
    main()
