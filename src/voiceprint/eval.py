#!/usr/bin/env python3
"""Evaluation / inference script — test your LoRA adapter.

ELI5 — How do we use the trained adapter?
─────────────────────────────────────────

After training we have a small adapter file (usually < 50 MB).  To use
it we:
  1. Load the base model (can be full precision or quantized).
  2. Load the adapter on top with PEFT.
  3. Generate text — the model now has your "voice" baked in.

Later you'll want to:
  - Merge the adapter into the base model (saves one load step).
  - Convert to GGUF format for llama.cpp (see evals/README.md).
  - Test with `llama-cli` or `llama-server`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from voiceprint.config import BASE_MODEL_ID, TrainingConfig

console = Console()
train_cfg = TrainingConfig()

# Path to the saved adapter (output of train.py).
ADAPTER_DIR = train_cfg.output_dir


def load_merged_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model + LoRA adapter merged together.

    ELI5: Merging means we take the LoRA weights (B @ A) and add them
    directly into the base model's weight matrices.  The result is a
    single model file — no adapter needed at inference time.

    This is slower to set up (one-time cost) but faster for inference
    and required for llama.cpp conversion.
    """
    console.print(Panel.fit(
        "[bold cyan]Loading model + adapter[/bold cyan]",
        subtitle=f"Base: {BASE_MODEL_ID} | Adapter: {ADAPTER_DIR}",
    ))

    # Load base model at full precision for merging (better quality).
    # For inference on a 5090 you could also load at 4-bit to save memory.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
    )

    # Stack the LoRA adapter on top.
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    # Merge LoRA weights into the base model.
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR),
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]Model + adapter loaded (merged)[/green]")
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
