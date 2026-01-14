# vLLM Deployment Guide for RedSage

This guide covers deploying RedSage models using [vLLM](https://github.com/vllm-project/vllm), a high-throughput and memory-efficient inference engine with OpenAI-compatible API support.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Server Setup](#basic-server-setup)
- [Configuration Options](#configuration-options)
- [Client Examples](#client-examples)
- [Production Deployment](#production-deployment)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

> **Prerequisites:** Install vLLM first (see [Installation](#installation) section below).

```bash
# Start the server
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO --port 8000

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RISys-Lab/RedSage-Qwen3-8B-DPO",
    "messages": [
      {"role": "system", "content": "You are RedSage, a helpful cybersecurity assistant."},
      {"role": "user", "content": "What is SSRF?"}
    ]
  }'
```

---

## Installation

The commands below follow the official vLLM install guides.

### Prerequisites

- **Python**: 3.10–3.13 (officially tested versions)
- **OS**: Linux; Windows users should run vLLM inside WSL2
- **GPU**: NVIDIA GPU with compute capability ≥7.0; modern drivers required
- **CUDA**: Bundled via the PyTorch wheels selected by vLLM (recent wheels target CUDA 12.x); separate CUDA installs are not needed when using the recommended commands

### Install vLLM (recommended)

```bash
# Create and seed an env (example with uv)
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vLLM and let it pick the right torch backend (CUDA/CPU)
uv pip install vllm --torch-backend=auto
```

### Alternative (pip-only)

```bash
pip install vllm --torch-backend=auto
```

### Verify Installation

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## Basic Server Setup

### Starting a Server

The simplest way to start a vLLM server for RedSage:

```bash
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --port 8000 \
  --host 0.0.0.0
```

This launches an OpenAI-compatible API server at `http://localhost:8000/v1`.

> vLLM will load generation defaults from a model's `generation_config.json` (if present). Override any setting at serve time with CLI flags (e.g., `--max-model-len`, `--temperature`) or a custom config via `--generation-config`.

### Model Variants

RedSage offers multiple model variants:

```bash
# Instruction-tuned model (recommended for most use cases)
vllm serve RISys-Lab/RedSage-Qwen3-8B-Ins --port 8000

# DPO-aligned model (preference-optimized responses)
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO --port 8000

# Base model (for fine-tuning or research)
vllm serve RISys-Lab/RedSage-Qwen3-8B-Base --port 8000
```

> **Note:** The `-Ins` and `-DPO` variants are chat-aligned models. Do not use `-Base` for direct chat inference without fine-tuning.

---

## Configuration Options

### Context Length

Control the maximum context window size:

```bash
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --max-model-len 32768 \
  --port 8000
```

Choose a context length that fits your GPU memory budget; higher limits require more VRAM.

### Multi-GPU Deployment

Distribute the model across multiple GPUs using tensor parallelism:

```bash
# Use 2 GPUs
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --tensor-parallel-size 2 \
  --port 8000

# Use 4 GPUs
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --tensor-parallel-size 4 \
  --port 8000
```

**GPU Requirements:**
- 1 GPU: 16GB+ VRAM (A10, A100, H100, RTX 4090, etc.)
- 2 GPUs: 2×16GB or 2×24GB for extended context
- 4 GPUs: For maximum throughput or very long contexts

### Data Type and Precision

Control the precision for performance/memory trade-offs:

```bash
# BFloat16 (recommended for A100, H100)
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --dtype bfloat16 \
  --port 8000

# Float16 (for older GPUs)
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --dtype float16 \
  --port 8000

# Auto-detect best dtype
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --dtype auto \
  --port 8000
```

### Trust Remote Code

Some models may require remote code execution:

```bash
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --trust-remote-code \
  --port 8000
```

### Complete Example

A production-ready configuration:

```bash
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16
```

---

## Client Examples

### Python (OpenAI SDK)

Install the OpenAI Python client:

```bash
pip install openai
```

**Example code:**

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # vLLM doesn't require API key for local servers
)

# Chat completion
response = client.chat.completions.create(
    model="RISys-Lab/RedSage-Qwen3-8B-DPO",
    messages=[
        {"role": "system", "content": "You are RedSage, a helpful cybersecurity assistant."},
        {"role": "user", "content": "Explain SQL injection and how to prevent it."}
    ],
    temperature=0.2,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

**For a complete interactive demo, see:**
- `inference/vllm_demo.py` in this repository

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RISys-Lab/RedSage-Qwen3-8B-DPO",
    "messages": [
      {"role": "system", "content": "You are RedSage, a helpful cybersecurity assistant."},
      {"role": "user", "content": "What are the OWASP Top 10?"}
    ],
    "temperature": 0.2,
    "max_tokens": 512
  }'
```

### Node.js

Install the OpenAI Node.js client:

```bash
npm install openai
```

**Example code:**

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'EMPTY',
});

async function chat() {
  const response = await client.chat.completions.create({
    model: 'RISys-Lab/RedSage-Qwen3-8B-DPO',
    messages: [
      { role: 'system', content: 'You are RedSage, a helpful cybersecurity assistant.' },
      { role: 'user', content: 'What is XSS?' }
    ],
    temperature: 0.2,
    max_tokens: 512,
  });
  
  console.log(response.choices[0].message.content);
}

chat();
```

### LangChain

```python
# For LangChain >=0.1.0 with langchain-openai package
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="EMPTY",
    model_name="RISys-Lab/RedSage-Qwen3-8B-DPO",
    temperature=0.2,
)

messages = [
    SystemMessage(content="You are RedSage, a helpful cybersecurity assistant."),
    HumanMessage(content="Explain buffer overflow attacks."),
]

response = llm(messages)
print(response.content)
```

---

## Production Deployment

### Docker Deployment

**Dockerfile:**

```dockerfile
FROM vllm/vllm-openai:latest

ENV MODEL_NAME=RISys-Lab/RedSage-Qwen3-8B-DPO
ENV HOST=0.0.0.0
ENV PORT=8000

# Use exec form with sh -c for proper signal handling and variable substitution
CMD ["sh", "-c", "vllm serve ${MODEL_NAME} --host ${HOST} --port ${PORT}"]
```

**Build and run:**

```bash
docker build -t redsage-vllm .
docker run --gpus all -p 8000:8000 redsage-vllm
```

**Override environment variables at runtime:**

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=RISys-Lab/RedSage-Qwen3-8B-Ins \
  -e PORT=8001 \
  redsage-vllm
```

### System Service (systemd)

Create `/etc/systemd/system/redsage-vllm.service`:

```ini
[Unit]
Description=RedSage vLLM Server
After=network.target

[Service]
Type=simple
User=vllm
WorkingDirectory=/opt/redsage
ExecStart=/usr/local/bin/vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --tensor-parallel-size 2
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable redsage-vllm
sudo systemctl start redsage-vllm
sudo systemctl status redsage-vllm
```

---

## Performance Tuning

- Adjust `--max-model-len` to fit GPU memory; longer contexts use more VRAM.
- Use `--tensor-parallel-size N` to shard across multiple GPUs for larger context or throughput.
- Select a dtype suitable for your hardware: `--dtype bfloat16` on modern GPUs, `--dtype float16` otherwise.
- Choose an attention backend if needed (e.g., `--attention-backend FLASH_ATTN` on CUDA builds) based on the official backend list.

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA OOM errors during startup or inference.

**Solutions:**

1. Reduce context length:
   ```bash
   vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO --max-model-len 4096
   ```

2. Use tensor parallelism across multiple GPUs:
   ```bash
   vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO --tensor-parallel-size 2
   ```

### Slow Inference

**Symptoms:** High latency per request.

**Solutions:**

1. Enable batching (if not already):
  Ensure your client batches requests when possible.

2. Use appropriate data type:
   ```bash
   vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO --dtype bfloat16
   ```

3. Increase GPU resources (more GPUs, tensor parallelism).

### Connection Refused

**Symptoms:** Cannot connect to vLLM server.

**Solutions:**

1. Verify server is running:
   ```bash
   ps aux | grep vllm
   ```

2. Check if port is bound:
   ```bash
   netstat -tuln | grep 8000
   ```

3. Ensure host is set to `0.0.0.0` for external access:
   ```bash
   vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO --host 0.0.0.0 --port 8000
   ```

### Model Loading Failures

**Symptoms:** Errors during model download or loading.

**Solutions:**

1. Check Hugging Face credentials:
   ```bash
   huggingface-cli login
   ```

2. Manually download the model:
   ```bash
   huggingface-cli download RISys-Lab/RedSage-Qwen3-8B-DPO
   ```

3. Specify local model path:
   ```bash
   vllm serve /path/to/local/model --port 8000
   ```

### API Compatibility Issues

**Symptoms:** Client libraries fail to connect or receive unexpected responses.

**Solutions:**

1. Verify OpenAI client version:
   ```bash
   pip install --upgrade openai
   ```

2. Check API endpoint format (should be `/v1/chat/completions`).

3. Ensure model name in request matches server model name exactly.

---

## Advanced Topics

### Custom Chat Templates

Override the tokenizer chat template if you need a different prompt format:

```bash
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO \
  --chat-template /path/to/custom_template.jinja \
  --port 8000
```

---

## Resources

- **vLLM Documentation:** https://docs.vllm.ai/
- **vLLM GitHub:** https://github.com/vllm-project/vllm
- **OpenAI API Reference:** https://platform.openai.com/docs/api-reference
- **RedSage Models:** https://huggingface.co/RISys-Lab

---

## Support

For issues specific to:
- **RedSage models:** Open an issue in the [RedSage GitHub repository](https://github.com/RISys-Lab/RedSage)
- **vLLM deployment:** Check the [vLLM GitHub issues](https://github.com/vllm-project/vllm/issues)

---

**Last Updated:** December 2025
