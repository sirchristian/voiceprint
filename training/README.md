# Training

The main training script is `src/voiceprint/train.py`.  This directory is reserved for training artifacts, notebooks, and experiment logs.

## Running training

```bash
uv run python -m voiceprint.train
```

## Adjusting settings

Edit `src/voiceprint/config.py` — every parameter has an ELI5 docstring.

## VRAM guide (RTX 5090, 32 GB)

| Setting | Low VRAM | Default | Max Quality |
|---------|----------|---------|-------------|
| Batch size | 2 | 4 | 8 |
| Seq length | 256 | 512 | 1024 |
| Gradient checkpointing | on | on | off |
| Optim | paged_adamw_8bit | paged_adamw_8bit | adamw_torch |
