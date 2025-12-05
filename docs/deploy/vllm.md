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

```bash
# Install vLLM
pip install vllm

# Start the server
vllm serve RISys-Lab/RedSage-8B-DPO --port 8000

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RISys-Lab/RedSage-8B-DPO",
    "messages": [
      {"role": "system", "content": "You are RedSage, a helpful cybersecurity assistant."},
      {"role": "user", "content": "What is SSRF?"}
    ]
  }'
```

---

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **GPU**: NVIDIA GPU with at least 16GB VRAM for 8B models
- **Operating System**: Linux (recommended) or Windows with WSL2

### Install vLLM

```bash
# Install via pip (recommended)
pip install vllm

# Or install from source for latest features
pip install git+https://github.com/vllm-project/vllm.git
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
vllm serve RISys-Lab/RedSage-8B-DPO \
  --port 8000 \
  --host 0.0.0.0
```

This launches an OpenAI-compatible API server at `http://localhost:8000/v1`.

### Model Variants

RedSage offers multiple model variants:

```bash
# Instruction-tuned model (recommended for most use cases)
vllm serve RISys-Lab/RedSage-8B-Ins --port 8000

# DPO-aligned model (preference-optimized responses)
vllm serve RISys-Lab/RedSage-8B-DPO --port 8000

# Base model (for fine-tuning or research)
vllm serve RISys-Lab/RedSage-8B-Base --port 8000
```

> **Note:** The `-Ins` and `-DPO` variants are chat-aligned models. Do not use `-Base` for direct chat inference without fine-tuning.

---

## Configuration Options

### Context Length

Control the maximum context window size:

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --max-model-len 32768 \
  --port 8000
```

**Recommendations:**
- Default: 8192 tokens (if not specified)
- Extended: 32768 tokens (for long documents/conversations)
- Memory-constrained: 4096 tokens

### Multi-GPU Deployment

Distribute the model across multiple GPUs using tensor parallelism:

```bash
# Use 2 GPUs
vllm serve RISys-Lab/RedSage-8B-DPO \
  --tensor-parallel-size 2 \
  --port 8000

# Use 4 GPUs
vllm serve RISys-Lab/RedSage-8B-DPO \
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
vllm serve RISys-Lab/RedSage-8B-DPO \
  --dtype bfloat16 \
  --port 8000

# Float16 (for older GPUs)
vllm serve RISys-Lab/RedSage-8B-DPO \
  --dtype float16 \
  --port 8000

# Auto-detect best dtype
vllm serve RISys-Lab/RedSage-8B-DPO \
  --dtype auto \
  --port 8000
```

### Batch Processing

Increase throughput with request batching:

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --port 8000
```

### Trust Remote Code

Some models may require remote code execution:

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --trust-remote-code \
  --port 8000
```

### Complete Example

A production-ready configuration:

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-num-seqs 256 \
  --disable-log-requests
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
    model="RISys-Lab/RedSage-8B-DPO",
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
    "model": "RISys-Lab/RedSage-8B-DPO",
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
    model: 'RISys-Lab/RedSage-8B-DPO',
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
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="EMPTY",
    model_name="RISys-Lab/RedSage-8B-DPO",
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

ENV MODEL_NAME=RISys-Lab/RedSage-8B-DPO
ENV HOST=0.0.0.0
ENV PORT=8000

# Use shell form to allow variable substitution
CMD vllm serve ${MODEL_NAME} --host ${HOST} --port ${PORT}
```

**Build and run:**

```bash
docker build -t redsage-vllm .
docker run --gpus all -p 8000:8000 redsage-vllm
```

### Kubernetes Deployment

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redsage-vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redsage-vllm
  template:
    metadata:
      labels:
        app: redsage-vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - --model
          - RISys-Lab/RedSage-8B-DPO
          - --host
          - "0.0.0.0"
          - --port
          - "8000"
          - --max-model-len
          - "32768"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: redsage-vllm-service
spec:
  selector:
    app: redsage-vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### Reverse Proxy (Nginx)

Enable request batching and load balancing with Nginx:

```nginx
upstream vllm_backend {
    server localhost:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    location /v1/ {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}
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
ExecStart=/usr/local/bin/vllm serve RISys-Lab/RedSage-8B-DPO \
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

### Memory Optimization

**Reduce KV cache size:**

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

**Enable CPU offloading (for limited VRAM):**

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --cpu-offload-gb 16 \
  --port 8000
```

### Throughput Optimization

**Increase batch size:**

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --port 8000
```

**Enable speculative decoding (experimental):**

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --speculative-model <smaller-model> \
  --num-speculative-tokens 5 \
  --port 8000
```

### Latency Optimization

**Reduce batch size for lower latency:**

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --max-num-seqs 64 \
  --port 8000
```

**Disable continuous batching:**

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --disable-frontend-multiprocessing \
  --port 8000
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA OOM errors during startup or inference.

**Solutions:**

1. Reduce context length:
   ```bash
   vllm serve RISys-Lab/RedSage-8B-DPO --max-model-len 4096
   ```

2. Lower GPU memory utilization:
   ```bash
   vllm serve RISys-Lab/RedSage-8B-DPO --gpu-memory-utilization 0.8
   ```

3. Use tensor parallelism across multiple GPUs:
   ```bash
   vllm serve RISys-Lab/RedSage-8B-DPO --tensor-parallel-size 2
   ```

### Slow Inference

**Symptoms:** High latency per request.

**Solutions:**

1. Enable batching (if not already):
   ```bash
   vllm serve RISys-Lab/RedSage-8B-DPO --max-num-seqs 128
   ```

2. Use appropriate data type:
   ```bash
   vllm serve RISys-Lab/RedSage-8B-DPO --dtype bfloat16
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
   vllm serve RISys-Lab/RedSage-8B-DPO --host 0.0.0.0 --port 8000
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
   huggingface-cli download RISys-Lab/RedSage-8B-DPO
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

### Quantization

For reduced memory footprint, use quantized models (when available):

```bash
# AWQ quantization (4-bit)
vllm serve RISys-Lab/RedSage-8B-DPO-AWQ --quantization awq --port 8000

# GPTQ quantization (4-bit)
vllm serve RISys-Lab/RedSage-8B-DPO-GPTQ --quantization gptq --port 8000
```

> **Note:** Quantized variants will be released separately. Check the model collection on Hugging Face.

### Custom Chat Templates

Override the default chat template:

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --chat-template /path/to/custom_template.jinja \
  --port 8000
```

### Monitoring and Metrics

Enable Prometheus metrics:

```bash
vllm serve RISys-Lab/RedSage-8B-DPO \
  --enable-metrics \
  --port 8000
```

Access metrics at `http://localhost:8000/metrics`.

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
