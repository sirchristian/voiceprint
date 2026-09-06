#!/usr/bin/env python3
"""Evaluation / inference script — test your LoRA adapter.

ELI5 — How do we use the trained adapter?
─────────────────────────────────────────

After training we have a small adapter file (usually < 50 MB).  To use
it we:
  1. Load the base model (can be full precision or quantized).
  2. Load the adapter on top with PEFT.
  3. Generate text — the model now has your "voice" baked in.

Chat models use a "chat template" that wraps the user's prompt in
special tokens (like user / </think>

) so the model
knows who is speaking.  We must use the same template at inference
that we used at training time, or the model won't understand the input.

You can also load the model directly from MLFlow:
    import mlflow.pytorch
    model = mlflow.pytorch.load_model("runs:/<run_id>/model")

Later you'll want to:
  - Merge the adapter into the base model (saves one load step).
  - Convert to GGUF format for llama.cpp (see evals/README.md).
  - Test with `llama-cli` or `llama-server`.
"""

from __future__ import annotations

import argparse
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
        enable_thinking=False,  # Disable Qwen3 thinking tags at inference too.
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

    # Load tokenizer with thinking disabled so it doesn't generate
    # <thinking>...</thinking> blocks during inference.
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR),
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
        enable_thinking=False,
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
    """Generate a chat response from a user prompt.

    ELI5: Instead of just tokenizing the raw prompt, we wrap it in the
    model's chat template.  This adds the special tokens that tell the
    model "this is what the user said, now you reply as assistant."

    The chat template for Qwen3 looks like:
         $user
        {prompt}

        $assistant
        {thinking tags if enabled}

    We also use `add_generation_prompt=True` which adds the assistant's
    opening tags so the model knows to start generating its reply.

    Temperature controls randomness: 0.0 = most likely token every time,
    higher = more creative but potentially nonsensical.  For a "voice"
    model you might want 0.7-0.9 to sound natural.
    """
    # Wrap the prompt in the chat template.
    # The `messages` list is a conversation history — here just one turn.
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Add the assistant's opening tags.
        tokenize=True,               # Return token IDs, not raw text.
        return_tensors="pt",        # Return tensors ready for model.generate().
    )
    inputs = input_ids.to(model.device)

    with torch.no_grad():  # No gradients needed for inference.
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,             # Nucleus sampling — ignore unlikely tokens.
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt).
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def load_base_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load only the base model (no LoRA adapter) for comparison.

    ELI5: Sometimes you want to see what the base model says *before*
    your LoRA adapter is applied.  This lets you compare:
      - Base model response (vanilla, unmodified model)
      - LoRA response (base model + your voice adapter)
    So you can tell if the adapter is actually doing something.
    """
    console.print(Panel.fit(
        "[bold cyan]Loading base model only (no adapter)[/bold cyan]",
        subtitle=f"Model: {BASE_MODEL_ID}",
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

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
        enable_thinking=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]Base model loaded (4-bit, no adapter)[/green]")
    return model, tokenizer


def quick_test(base_only: bool = False) -> None:
    """Run a few sample prompts to see if the model sounds right.

    Args:
        base_only: If True, load only the base model (no LoRA adapter).
    """
    if base_only:
        model, tokenizer = load_base_model()
        label = "[dim]BASE[/dim]"
    else:
        model, tokenizer = load_merged_model()
        label = "[dim]LoRA[/dim]"

    prompts = [
        "Write a quick email asking whether there is a recording.",
        "Review this approach and tell me whether you'd merge it.",
    ]

    for prompt in prompts:
        console.print(f"\n[dim]Prompt:[/dim] {prompt}")
        response = generate(model, tokenizer, prompt)
        console.print(f"{label} Response: [green]{response}[/green]")


def main() -> None:
    """Entry point for evaluation."""
    parser = argparse.ArgumentParser(description="voiceprint eval")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Load only the base model (no LoRA adapter) for comparison.",
    )
    args = parser.parse_args()

    if not args.base_only and not ADAPTER_DIR.exists():
        console.print(f"[bold red]Adapter not found at {ADAPTER_DIR}[/bold red]")
        console.print("[dim]Run training first: uv run python -m voiceprint.train[/dim]")
        raise SystemExit(1)

    console.print(Panel(
        "[bold yellow]voiceprint[/bold yellow] — Eval / Inference\n"
        f"{'Base model only (no adapter)' if args.base_only else f'Adapter: {ADAPTER_DIR}'}",
        title="🎙️ VoicePrint Eval",
    ))

    quick_test(base_only=args.base_only)


if __name__ == "__main__":
    main()
