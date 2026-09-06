"""Data loading and preprocessing.

ELI5 — What happens to our text before it hits the model?
─────────────────────────────────────────────────────────

SFTTrainer (from the TRL library) handles tokenization internally.  We
just need to load the raw text into a HuggingFace Dataset with a
"text" column, and SFTTrainer takes care of:

  1. Tokenizing each example with the model's tokenizer.
  2. Truncating to max_seq_length.
  3. Creating the labels (input_ids shifted by one for causal LM).
  4. Padding and batching.

All we do here is load the JSONL and return the dataset.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

from voiceprint.config import DataConfig


def load_and_prepare(
    data_cfg: DataConfig,
) -> Dataset:
    """Load a JSONL file and return a HuggingFace Dataset.

    Args:
        data_cfg:   Paths to the JSONL data files.

    Returns:
        A Dataset with a single column: {"text": str}.
        SFTTrainer will tokenize it internally.
    """
    # Load from local JSONL.  Each line should be {"text": "..."}
    raw = load_dataset("json", data_files={"train": str(data_cfg.train_file)})["train"]

    return raw


def print_dataset_stats(dataset: Dataset) -> None:
    """Print a quick summary of the dataset for debugging."""
    print(f"\n  Dataset: {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")
    if "input_ids" in dataset.column_names:
        sample = dataset[0]["input_ids"]
        print(f"  Sample input_ids length: {len(sample)} (first 10: {sample[:10]})")
