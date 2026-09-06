"""Data loading and preprocessing.

ELI5 — What happens to our text before it hits the model?
─────────────────────────────────────────────────────────

Chat models expect conversations, not raw text. The tokenizer has a
"chat template" that wraps messages in special tokens (like
<|im_start|>user / <|im_start|>assistant) so the model learns the
pattern of "user says X, assistant replies Y".

Two approaches exist:
  1. Pre-apply the chat template ourselves (this file).  Simple,
     explicit, works with any model.
  2. Let SFTTrainer handle it via a `messages` column.  Works in theory
     but is framework-dependent and harder to debug.

We go with option 1: load `{"messages": [...]}` JSONL, apply the
tokenizer's chat template to get a `text` column, and let SFTTrainer
handle the rest (tokenization, batching, truncation).

The chat template call also supports `enable_thinking=False` for Qwen3
models, which strips the <thinking> tags from training data so the model
learns to respond directly instead of overthinking.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from voiceprint.config import BASE_MODEL_ID, DataConfig


def load_and_prepare(
    data_cfg: DataConfig,
    model_id: str = BASE_MODEL_ID,
) -> Dataset:
    """Load a JSONL file, apply the model's chat template, return a HF Dataset.

    Expected JSONL format (one line per example):
        {"messages": [
            {"role": "user", "content": "Write a quick email..."},
            {"role": "assistant", "content": "Sure, here's a draft..."}
        ]}

    The chat template converts the message list into a single string like:
        <|im_start|>user\nWrite a quick email...<|im_end|>\n<|im_start|>assistant\nSure...<|im_end|>

    ELI5: The chat template is like a stage director — it adds the
    "stage directions" (special tokens) that tell the model who is
    speaking and when to stop.  Without it, the model just sees raw
    text and doesn't know the conversation structure.

    Args:
        data_cfg:  Paths to the JSONL data files.
        model_id:  Model ID for the tokenizer (defaults to BASE_MODEL_ID).

    Returns:
        A Dataset with a single column: {"text": str}.
        SFTTrainer will tokenize it internally.
    """
    # Load from local JSONL.
    raw = load_dataset("json", data_files={"train": str(data_cfg.train_file)})["train"]

    # Load the tokenizer to apply the chat template.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        # Qwen3-specific: disable thinking tags in training data.
        # Without this, the template adds <thinking>...</thinking> blocks
        # and the model learns to overthink every response.
        enable_thinking=False,
    )

    def apply_template(example: dict[str, list[dict]]) -> dict[str, str]:
        """Convert a message list into a single chat-formatted string."""
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                # enable_thinking is already set on the tokenizer,
                # but being explicit doesn't hurt.
                add_generation_prompt=False,  # We're training, not generating.
            ),
        }

    # Quick format check — catch the old {"text": "..."} format early.
    first = raw[0]
    if "messages" not in first:
        raise ValueError(
            f"Expected 'messages' key in JSONL data, got {list(first.keys())}.\n"
            "Your data file uses the old {{'text': '...'}} format.\n"
            "See data/README.md for the current format (messages array)."
        )

    # Map each example through the template.
    dataset = raw.map(apply_template, remove_columns=["messages"])

    return dataset


def print_dataset_stats(dataset: Dataset) -> None:
    """Print a quick summary of the dataset for debugging."""
    print(f"\n  Dataset: {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")
    if len(dataset) > 0:
        sample = dataset[0]["text"]
        print(f"  Sample length: {len(sample)} chars")
        print(f"  Sample preview: {sample[:120]}...")
