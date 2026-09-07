#!/usr/bin/env python3
"""Merge a trained LoRA adapter and prepare a llama.cpp GGUF export.

ELI5 — Why this file exists:
────────────────────────────
The adapter we save after fine-tuning is tiny and easy to swap in.  But
llama.cpp wants a single standalone model file, usually in GGUF format.
So the flow is:

    1. Load the base model.
    2. Merge the trained LoRA weights into it.
    3. Save the merged model to a local directory.
    4. Run llama.cpp's `convert_hf_to_gguf.py` on that directory.

This keeps the workflow explicit and avoids pretending the LoRA directory
is the final runtime artifact for llama.cpp.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from voiceprint.config import BASE_MODEL_ID, TrainingConfig

train_cfg = TrainingConfig()


def merge_adapter_into_base(
    base_model_id: str = BASE_MODEL_ID,
    adapter_dir: str | Path = train_cfg.output_dir,
    output_dir: str | Path = "models/user-voice-merged",
) -> Path:
    """Merge a trained LoRA adapter into the base model.

    The resulting folder is a normal Hugging Face model directory that the
    llama.cpp conversion script can consume.
    """
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(Path("models").resolve()),
    )

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir))

    # The GGUF converter needs the tokenizer vocabulary and config alongside
    # the weights; the adapter directory already contains the correct files.
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir),
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(output_dir))

    return output_dir


def build_llama_cpp_command(
    model_dir: str | Path,
    output_dir: str | Path,
    outfile: str,
    outtype: str = "q4_k_m",
    llama_cpp_dir: str | Path = "/opt/llama.cpp",
) -> str:
    """Build the command needed to run llama.cpp's GGUF converter."""
    llama_cpp_dir = Path(llama_cpp_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile_path = output_dir / outfile
    script_path = llama_cpp_dir / "convert_hf_to_gguf.py"

    command = [
        "uv",
        "run",
        "python",
        str(script_path),
        str(model_dir),
        "--outfile",
        str(outfile_path),
        "--outtype",
        outtype,
        # Merged model has no MTP/NextN weights (merge_and_unload only
        # produces the main decoder layers), but the base config.json
        # still advertises them — drop that stale metadata or llama.cpp
        # looks for a blk.N.attn_norm.weight that was never written.
        "--no-mtp",
    ]
    return shlex.join(command)


def main() -> None:
    """CLI entry point for merging and preparing the GGUF export command."""
    parser = argparse.ArgumentParser(description="Merge a voiceprint adapter and prepare a GGUF export.")
    parser.add_argument("--adapter-dir", default=str(train_cfg.output_dir), help="Directory containing the LoRA adapter.")
    parser.add_argument("--merged-dir", default="models/user-voice-merged", help="Where to save the merged Hugging Face model.")
    parser.add_argument("--output-dir", default="models/user-voice-gguf", help="Where to write the GGUF file.")
    parser.add_argument("--outfile", default="user-voice-f16.gguf", help="Intermediate GGUF filename to write.")
    parser.add_argument("--outtype", default="f16", help="Converter output type; use f16 before llama-quantize.")
    parser.add_argument("--llama-cpp-dir", default="/opt/llama.cpp", help="Local path to the llama.cpp checkout.")
    args = parser.parse_args()

    merged_dir = merge_adapter_into_base(
        adapter_dir=args.adapter_dir,
        output_dir=args.merged_dir,
    )

    command = build_llama_cpp_command(
        model_dir=merged_dir,
        output_dir=args.output_dir,
        outfile=args.outfile,
        outtype=args.outtype,
        llama_cpp_dir=args.llama_cpp_dir,
    )

    print(f"Merged model saved to: {merged_dir}")
    print(f"Run this to convert to GGUF:\n{command}")


if __name__ == "__main__":
    main()