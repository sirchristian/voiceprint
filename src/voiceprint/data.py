"""Data loading and preprocessing.

ELI5 — What happens to our text before it hits the model?
─────────────────────────────────────────────────────────

Raw text → Token IDs → Padded batches

1. The tokenizer converts each string into a list of integer IDs.
   (Think of it as translating English into the model's native language.)

2. We truncate long sequences to `max_seq_length` tokens and pad short
   ones so every example in a batch has the same length.

3. For causal language modeling (what we're doing) the labels are the
   same as the input — the model tries to predict each token given all
   previous tokens.  The cross-entropy loss is computed against these
   labels.

The `datasets` library (from HuggingFace) handles most of the heavy
lifting.  We just wire it up with the right tokenizer and config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from jaxtyping import Float  # noqa: F401 — used for type hints
from transformers import AutoTokenizer

from voiceprint.config import DataConfig, TrainingConfig


def load_and_prepare(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    model_id: str,
) -> Dataset:
    """Load a JSONL file and return a prepared HuggingFace Dataset.

    Args:
        data_cfg:   Paths to the JSONL data files.
        train_cfg:  Training config (for max_seq_length).
        model_id:   HuggingFace model ID for loading the matching tokenizer.

    Returns:
        A Dataset with columns: {"text": str, "input_ids": list[int], "attention_mask": list[int]}.
    """
    # Load from local JSONL.  Each line should be {"text": "..."}
    raw = load_dataset("json", data_files={"train": str(data_cfg.train_file)})["train"]

    # Grab the tokenizer that matches our base model.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,  # Some models (Qwen) need this flag.
    )

    # Qwen tokenizer doesn't have a default pad token — set it explicitly.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        """Convert a single text example into token IDs."""
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=train_cfg.max_seq_length,
            padding="max_length",  # Pad to exactly max_seq_length
        )

    # Apply tokenization to the whole dataset.
    processed = raw.map(
        tokenize,
        remove_columns=["text"],  # We don't need the raw text after tokenizing.
        desc="Tokenizing",
    )

    # Rename to the names HuggingFace Trainer expects.
    # For causal LM, `labels` == `input_ids`.
    processed = processed.rename_column("input_ids", "input_ids")
    processed = processed.rename_column("attention_mask", "attention_mask")

    return processed


def print_dataset_stats(dataset: Dataset) -> None:
    """Print a quick summary of the dataset for debugging."""
    print(f"\n  Dataset: {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")
    if "input_ids" in dataset.column_names:
        sample = dataset[0]["input_ids"]
        print(f"  Sample input_ids length: {len(sample)} (first 10: {sample[:10]})")
