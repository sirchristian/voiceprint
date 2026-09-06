# Training Data

## Format

A single JSONL file where each line is one training example:

```jsonl
{"text": "Hey, what's up?\nI'm working on a new project. It's a QLoRA fine-tuning setup for a personal voice model.\nThat sounds cool, how's it going?"}
```

Each `text` field should be a **complete** example of your writing/voice — conversations, emails, code comments, essays, whatever represents how you communicate.

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
# Manual: just write a JSONL file with your text examples.
# Programmatic: use a script to extract from your writing history.
# See evals/ for extraction scripts.
```

## Validation split (optional)

If you have a separate validation file, set `DataConfig.val_file` in `config.py`.  The trainer will use it to track validation loss (helps detect overfitting).
