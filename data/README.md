# Training Data

## Format

A single JSONL file where each line is one training example:

```jsonl
{"messages": [
  {"role": "user", "content": "Write a quick email asking whether there is a recording."},
  {"role": "assistant", "content": "Sure, here's a draft:\n\nHey team — I missed standup..."}
]}
```

Each example is a **user prompt + your reply**.  The model learns the pattern of
"when someone asks this, you reply like that."

The model's chat template is applied automatically during loading, so you don't
need to know Qwen's special tokens (`user`,
</think>
) — just write plain JSON.

## Tips for a good voice dataset

### Quantity
- Start with **50-200 high-quality examples** for a tiny model like Qwen3-0.6B.
- For a 7B model aim for **500-2000 examples**.
- Quality > quantity: diverse, representative examples beat bulk.

### Diversity
Include different "modes" of your voice:
- **Casual conversation** (Discord, texts, chat)
- **Technical writing** (code comments, PR descriptions, docs)
- **Formal writing** (emails, proposals)
- **Creative writing** (if applicable)

### Structure
The model learns the patterns in your text.  Good patterns:
- How you start sentences
- Your punctuation style
- Your vocabulary choices
- How you explain things
- Your humor / tone

### Things to avoid
- Don't include sensitive data (passwords, keys, PII)
- Don't include copyrighted material you don't own
- Don't mix in data from other people's voices (confuses the model — keep it to one voice per run)

## Creating the file

```bash
# Copy the example and fill in your own writing:
cp data/example.jsonl data/train.jsonl
# Then edit data/train.jsonl with your real examples.

# Programmatic: use a script to extract from your writing history
# (GitHub PRs, emails, Slack exports, etc.)
```

## Validation split (optional)

If you have a separate validation file, set `DataConfig.val_file` in `config.py`.  The trainer will use it to track validation loss (helps detect overfitting).
