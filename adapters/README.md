# Adapters

LoRA adapter checkpoints live here.  Each training run produces a folder like:

```
adapters/
└── user-voice-v1/           # Your adapter
    ├── adapter_model.safetensors    # The LoRA weights (~10-50 MB)
    ├── adapter_config.json           # LoRA hyperparameters used
    ├── tokenizer.json                # Tokenizer
    └── special_tokens_map.json
```

## What's inside an adapter?

- `adapter_model.safetensors` — The actual LoRA weights (tiny!)
- `adapter_config.json` — Records the LoRA rank, alpha, target modules
- Tokenizer files — Needed to encode/decode text

## Loading an adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
model = PeftModel.from_pretrained(model, "adapters/user-voice-v1")
```
