# RedSage: A Cybersecurity Generalist LLM

**Official repository for the RedSage paper (ICLR 2026 - Under Review).**

Authors: Naufal Suryanto, Muzammal Naseer, Pengfei Li, Syed Talal Wasim, Jinhui Yi, Juergen Gall, Paolo Ceravolo, Ernesto Damiani

<p align="center">
  🔧 <a href="https://huggingface.co/RISys-Lab">Hugging Face</a>&nbsp;&nbsp;|&nbsp;&nbsp;🧪 <a href="https://huggingface.co/collections/RISys-Lab/redsage-models">Model Collection</a>&nbsp;&nbsp;|&nbsp;&nbsp;📘 <a href="https://huggingface.co/collections/RISys-Lab/redsage-datasets">Data Collection</a>&nbsp;&nbsp;|&nbsp;&nbsp;📊 <a href="https://huggingface.co/collections/RISys-Lab/redsage-benchmarks">Benchmark Collection</a>&nbsp;&nbsp;|&nbsp;&nbsp;📑 <a href="https://openreview.net/forum?id=W4FAenIrQ2">Paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;🗞️ <a href="#">Blog (Comming Soon)</a>
</p>

<!-- 🖥️ <a href="https://huggingface.co/spaces/your-org/RedSage-Demo">Demo</a>&nbsp;&nbsp; -->

**TL;DR:** **RedSage** is a cybersecurity generalist LLM suite for diverse cybersecurity tasks. It combines domain-aware continual pretraining, agentic augmentation for SFT, and a dedicated benchmark (RedSage-Bench).

> Visit our HF org or the collection above for all checkpoints (names start with `RedSage-`).

---

## 📑 Table of Contents
- [News](#-news)
- [Introduction](#-introduction)
- [Run RedSage (Local)](#-run-redsage-local)
- [Deploy RedSage (vLLM)](#-deploy-redsage-vllm)
- [Build with RedSage](#-build-with-redsage)
- [Data](#-data)
- [Evaluation](#-evaluation)
- [Results Summary](#-results-summary)
- [Responsible Use & License](#-responsible-use--license)
- [Contributing](#-contributing)
- [Citation](#-citation)

## 📰 News
- 2026-01-13: Completed the lighteval implementation for RedSage-MCQ and related Cybersecurity benchmarks.
- 2025-10-14: Update the README.md
<!-- - 2025-08-12: Public release of **RedSage-Qwen3-8B-Base**, **-Ins**, **-DPO**.  
- 2025-08-05: RedSage-Bench task definitions added to `eval/lighteval_tasks/`.  
- 2025-07-28: Agentic augmentation pipeline open-sourced in `data/augment/`. -->

### Release Plan & Checklist

We are releasing RedSage sequentially in four phases. Track progress here (we’ll keep this list updated).

<details>
  <summary><b>View checklist</b></summary>

#### 1) Model & Inference
- [ ] Publish `RedSage-Qwen3-8B-Base` on Hugging Face (weights + model card)
- [x] Publish `RedSage-Qwen3-8B-Ins` on Hugging Face (weights + model card)
- [x] Publish `RedSage-Qwen3-8B-DPO` on Hugging Face (weights + model card)
- [x] Publish `RedSage-Qwen3-8B-CFW` on Hugging Face (weights + model card)
- [ ] Publish `RedSage-Qwen3-8B-Seed` on Hugging Face (weights + model card)
- [ ] Provide `inference/hf_chat.py` (Transformers chat example)
- [ ] Provide `inference/vllm_demo.py` (simple client)
- [ ] Add **vLLM** serving guide in `docs/deploy/vllm.md`
- [ ] (Optional) Release quantized variants (GGUF/AWQ/GPTQ) & notes

#### 2) Data
- [ ] Release **RedSage-CFW** on Hugging Face (datasets + card)
- [ ] Release **RedSage-Seed** on Hugging Face (datasets + card)
- [ ] Release **RedSage-Conv** on Hugging Face (datasets + card)
- [ ] Release cybersecurity-filetering code.
- [ ] Release agentic data augmentation code for generating multi-turn conversation from seed.
- [ ] Add `data/README.md` (provenance, dedup, cleaning, TOS/licensing)

#### 3) Evaluation
- [x] Release **RedSage-MCQ** data and lighteval implementation
- [x] Release lighteval task implementations for related **Cybersecurity Benchmarks**
- [x] Provide `eval/run_lighteval.py` and example command lines
- [ ] Release **RedSage-OpenQA** data and lighteval implementation
- [ ] Publish baseline results (RedSage variants + common 8B baselines)
- [ ] Add results table/plots to **Docs**

#### 4) Training
- [ ] Add Axolotl **CPT** (continual pretraining) notes/configs in `training/configs/cpt/`
- [ ] Add Axolotl **SFT** config(s) in `training/configs/sft/`
- [ ] Add Axolotl **DPO** config(s) in `training/configs/dpo/`
- [ ] Provide `scripts/train_*.sh` runners + `accelerate` tips
- [ ] Document hardware requirements & expected throughput

</details>

---


## 🤖 Introduction

- **Focus:** Cybersecurity knowledge, skills, and tool use.  
- **Training pipeline:** Continual pretraining on a cybersecurity-filtered corpus → instruction fine-tuning with agentically generated multi-turn dialogues → preference alignment.  
- **Benchmark:** RedSage-Bench (MCQs + open-ended) to measure knowledge, skills, and tooling competency.  

### Model lineup (8B)

- **RedSage-Qwen3-8B-Base** ([🤗 Model Card](https://huggingface.co/RISys-Lab/RedSage-Qwen3-8B-Base))  
  Continual-pretrained on cybersecurity-filtered data with a small general replay to reduce forgetting. Raw, strong domain knowledge; no chat alignment.  
  *Best for:* research baselines, downstream fine-tuning (SFT/DPO), adapters, and custom domain adaptation.

- **RedSage-Qwen3-8B-Ins** ([🤗 Model Card](https://huggingface.co/RISys-Lab/RedSage-Qwen3-8B-Ins))  
  `Base` + supervised fine-tuning on RedSage-Conv (agentic, multi-turn cyber dialogues) and a targeted general SFT mix. Helpful, step-by-step answers without preference tuning.  
  *Best for:* day-to-day cybersecurity assistant usage, explanations, playbook authoring, code/snippet help.

- **RedSage-Qwen3-8B-DPO** ([🤗 Model Card](https://huggingface.co/RISys-Lab/RedSage-Qwen3-8B-DPO))  
  `Ins` + preference alignment with DPO for more preference-aligned responses.  
  *Best for:* end-user chat, production-style assistants where tone/format consistency matters.

> **Notes:** `-Ins` and `-DPO` are non-thinking chat models (no `<think>` blocks). Use `-Base` if you plan to continue training.

<details>
  <summary><b>Previous / Experimental Variants</b></summary>

- **RedSage-Qwen3-8B-CFW** ([🤗 Model Card](https://huggingface.co/RISys-Lab/RedSage-Qwen3-8B-CFW)) — CPT on cybersecurity-filtered web only (ablation).  
- **RedSage-Qwen3-8B-Seed** ([🤗 Model Card](https://huggingface.co/RISys-Lab/RedSage-Qwen3-8B-Seed)) — CPT on curated seed sources only (ablation).
</details>

---

## 💻 Run RedSage (Local)

### 🤗 Transformers (Ins/DPO)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "RISys-Lab/RedSage-Qwen3-8B-Ins"

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

messages = [
  {"role": "system", "content": "You are RedSage, a helpful cybersecurity assistant."},
  {"role": "user", "content": "List three SSRF mitigations."}
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=300, temperature=0.2)
print(tok.decode(out[0], skip_special_tokens=True))
````

> **Note:** `-Ins` / `-DPO` are non-thinking chat models; no `<think>` blocks.

---

## 🚀 Deploy RedSage (vLLM)

RedSage is production-ready with **vLLM** for high-throughput, OpenAI-compatible serving.

### Start a server

```bash
pip install -U vllm
vllm serve RISys-Lab/RedSage-Qwen3-8B-DPO --port 8000 --max-model-len 32768
# OpenAI-compatible API at http://localhost:8000/v1
```

### Call the API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RISys-Lab/RedSage-Qwen3-8B-DPO",
    "messages": [
      {"role": "system", "content": "You are RedSage, a helpful cybersecurity assistant."},
      {"role": "user",   "content": "Explain AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H."}
    ],
    "temperature": 0.2,
    "max_tokens": 512
  }'
```

### Tips

* Use `--tensor-parallel-size` for multi-GPU, `--max-model-len` for long contexts.
* Prefer BF16/FP16 on recent GPUs; quantized weights will be linked in the collection if provided.
* Enable request batching in your gateway (nginx/Envoy) for best throughput.

---

## 🛠️ Build with RedSage

### Continued Pre-training, Fine-tuning, and Preference Optimization (Axolotl)

See **`training/README.md`** for:

* CPT, SFT, & DPO workflows (Axolotl)
* Config references under `training/configs/`
* Hardware/memory notes and troubleshooting
* Example run scripts in `scripts/`

---

## 📂 Data

* **Cybersecurity-filtered corpus** with global dedup; includes a small general-domain replay to reduce forgetting.
* **RedSage-Seed:** curated Knowledge / Skills / Tools sources.
* **RedSage-Conv:** agentically generated, multi-turn, role-grounded dialogues with automatic validation.

Licenses and source notes are documented in `data/README.md`.

---

## 🧪 Evaluation

See **`eval/README.md`** for detailed instructions on:

* **RedSage-Bench:** 30K MCQs + 240 open-ended items with an LLM-as-judge rubric.
* **Cybersecurity Benchmarks:** **CTI-Bench**, **CyberMetric**, **SecBench**, **SecEval**, **SECURE**, **MMLU-CSec**.

### Quick Start

```bash
# List all available tasks
python eval/run_lighteval.py --list-tasks

# Run a single benchmark
python eval/run_lighteval.py vllm \
  --model RISys-Lab/RedSage-Qwen3-8B-DPO \
  --tasks cybermetrics:500

# Run multiple benchmarks
python eval/run_lighteval.py vllm \
  --model RISys-Lab/RedSage-Qwen3-8B-DPO \
  --tasks cybermetrics:500,mmlu:cs_security,secbench:mcq-en \
  --output-dir results/my_eval

# Run curated benchmarks (e.g, All RedSage-MCQs)
python eval/run_lighteval.py vllm \
  --model RISys-Lab/RedSage-Qwen3-8B-DPO \
  --tasks tasks/redsage_mcqs.txt \
  --output-dir results/redsage_mcq
```

For more examples and advanced usage, see **`eval/README.md`**.

---


## 📊 Results Summary

TBD.

---

## ⚖️ Responsible Use & License

This project contains offensive-security knowledge and is **released for research use only**. Apply appropriate safeguards before any operational deployment.

* **Code & weights:** TBD.
* **Data:** TBD.
* **Benchmarks:** TBD.

---

## 🤝 Contributing

We welcome issues and PRs! Please:

* Follow `CONTRIBUTING.md`.
* Avoid sensitive or proprietary data.
* Include configs to reproduce results for new benchmarks or training tweaks.

---

## 🧾 Citation

```bibtex
@inproceedings{anonymous2025redsage,
  title={RedSage: A Cybersecurity Generalist {LLM}},
  author={Anonymous},
  booktitle={Submitted to The Fourteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=W4FAenIrQ2},
  note={under review}
}
```

