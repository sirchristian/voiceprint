# Evals & Conversion

## Testing the adapter

```bash
# Run quick generation tests
uv run python -m voiceprint.eval
```

## Merging the adapter into the base model

After training, merge the LoRA weights into the base model for standalone use:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "adapters/user-voice-v1")
model = model.merge_and_unload()
model.save_pretrained("models/user-voice-merged")
```

## Converting to GGUF for llama.cpp

Once you have a merged model, convert it to GGUF format for llama.cpp:

```bash
# Install the conversion tool
pip install llama-cpp-python

# Convert to GGUF (Q4_K_M = 4-bit, good balance of quality/speed)
python -m llama_cpp.convert \
    --input-dir models/user-voice-merged \
    --output-dir models/user-voice-gguf \
    --outfile user-voice-q4.gguf \
    --outtype q4_k_m
```

Then run with llama.cpp:

```bash
./llama-cli -m models/user-voice-gguf/user-voice-q4.gguf \
    -p "Hey, what are you working on?" \
    -n 256 -t 8
```

### Quantization formats for llama.cpp

| Format | Quality | Size | Best for |
|--------|---------|------|----------|
| Q8_0   | Highest | ~8x base | Archival |
| Q6_K   | Very high | ~6x base | Best quality inference |
| Q4_K_M | High | ~4x base | **Default choice** |
| Q4_0   | Good | ~4x base | Smallest decent quality |
| Q2_K   | Okay | ~2x base | Very constrained RAM |

## Benchmarking

Track token generation speed:

```bash
./llama-bench -m models/user-voice-gguf/user-voice-q4.gguf
```

On the 5090 with GGML_CUDA, expect ~200-400 tok/s for a 0.6B model.
