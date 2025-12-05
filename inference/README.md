# RedSage Inference Examples

This directory contains example scripts for running inference with RedSage models.

## Scripts

### `hf_chat.py` - Transformers Chat Example

Interactive chat interface using the Hugging Face Transformers library. Ideal for local inference and testing.

**Requirements:**
```bash
pip install transformers torch accelerate
```

**Usage:**
```bash
# Basic usage with default model (RedSage-8B-Ins)
python hf_chat.py

# Specify a different model
python hf_chat.py --model RISys-Lab/RedSage-8B-DPO

# Adjust generation parameters
python hf_chat.py --max-tokens 512 --temperature 0.2

# Custom system prompt
python hf_chat.py --system-prompt "You are a penetration testing expert."
```

**Features:**
- Interactive chat mode
- Configurable model, temperature, and token limits
- Custom system prompts
- Support for GPU/CPU inference

---

### `vllm_demo.py` - vLLM Client Demo

Client for interacting with RedSage models served via vLLM's OpenAI-compatible API.

**Requirements:**
```bash
pip install openai
```

**Prerequisites:**
Start a vLLM server first:
```bash
pip install vllm
vllm serve RISys-Lab/RedSage-8B-DPO --port 8000
```

**Usage:**
```bash
# Connect to default local vLLM server
python vllm_demo.py

# Connect to custom server
python vllm_demo.py --base-url http://192.168.1.100:8000/v1

# Specify model name
python vllm_demo.py --model RISys-Lab/RedSage-8B-Ins

# Adjust generation parameters
python vllm_demo.py --max-tokens 1024 --temperature 0.5
```

**Features:**
- Runs example queries demonstrating cybersecurity use cases
- Interactive chat mode
- OpenAI-compatible API client
- Configurable server URL and parameters

---

## Comparison: Transformers vs vLLM

| Feature | `hf_chat.py` (Transformers) | `vllm_demo.py` (vLLM) |
|---------|----------------------------|----------------------|
| **Use Case** | Local testing, research | Production deployment |
| **Throughput** | Lower (single request) | High (batched requests) |
| **Memory** | Standard PyTorch | Optimized KV cache |
| **Setup** | Simple, direct | Requires server setup |
| **API** | Transformers native | OpenAI-compatible |
| **Best For** | Development, debugging | Production, high-load |

---

## Model Recommendations

- **RedSage-8B-Ins**: General-purpose cybersecurity assistant (recommended for most users)
- **RedSage-8B-DPO**: Preference-aligned responses for production chat applications
- **RedSage-8B-Base**: Research and fine-tuning (not chat-aligned)

---

## Additional Resources

- **vLLM Deployment Guide**: See `docs/deploy/vllm.md` for comprehensive deployment instructions
- **Model Cards**: Visit https://huggingface.co/RISys-Lab for full model documentation
- **Training**: See `training/README.md` for fine-tuning and adaptation

---

## Troubleshooting

### Transformers Script Issues

**Out of Memory:**
- Reduce `--max-tokens`
- Use `--device cpu` for CPU inference
- Close other GPU applications

**Model Loading Errors:**
```bash
# Login to Hugging Face if needed
huggingface-cli login
```

### vLLM Script Issues

**Connection Refused:**
- Verify vLLM server is running: `ps aux | grep vllm`
- Check server URL and port match your configuration
- Ensure `--host 0.0.0.0` was used when starting the server

**API Errors:**
- Verify OpenAI package is installed: `pip install openai`
- Check that model name in script matches server model name

---

## Examples

### Single Query with Transformers

Note: Run this from the `inference` directory.

```bash
cd inference
python -c "
import sys
sys.path.insert(0, '.')
from hf_chat import load_model, chat_single_turn

model, tokenizer = load_model('RISys-Lab/RedSage-8B-Ins')
response = chat_single_turn(
    model, tokenizer,
    'Explain SQL injection',
    'You are RedSage, a helpful cybersecurity assistant.'
)
print(response)
"
```

### Batch Processing with vLLM

For batch processing, integrate vLLM with your application using the OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

queries = [
    "What is XSS?",
    "Explain CSRF attacks.",
    "List OWASP Top 10."
]

for query in queries:
    response = client.chat.completions.create(
        model="RISys-Lab/RedSage-8B-DPO",
        messages=[
            {"role": "system", "content": "You are RedSage, a helpful cybersecurity assistant."},
            {"role": "user", "content": query}
        ]
    )
    print(f"Q: {query}")
    print(f"A: {response.choices[0].message.content}\n")
```

---

**Need help?** Open an issue in the [RedSage GitHub repository](https://github.com/RISys-Lab/RedSage/issues).
