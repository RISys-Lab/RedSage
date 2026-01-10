# Evaluation

This directory contains evaluation code and benchmarks for RedSage models on cybersecurity tasks.

## Table of Contents

- [Setup](#setup)
- [Quick Start](#quick-start)
- [Running Evaluations](#running-evaluations)
- [Available Tasks](#available-tasks)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Setup

### 1. Environment

Create and activate the conda environment:

```bash
conda create -n lighteval python=3.10
conda activate lighteval
```

### 2. Install Lighteval

This repository uses `lighteval` as a submodule to ensure compatibility with our custom tasks.

```bash
# Initialize the submodule
git submodule update --init --recursive

# Install lighteval in editable mode
cd eval/lighteval
pip install -e .
cd ../../
```

### 3. Install Dependencies

Install additional dependencies required for the benchmarks:

```bash
pip install cvss aenum
```

## Quick Start

We provide a convenient wrapper script `run_lighteval.py` for easy evaluation:

```bash
# List all available tasks
python eval/run_lighteval.py --list-tasks

# Run a single task with Transformers backend
python eval/run_lighteval.py --model RISys-Lab/RedSage-8B-Ins --tasks cybermetrics:80

# Run with vLLM backend (recommended for faster inference)
python eval/run_lighteval.py vllm --model RISys-Lab/RedSage-8B-Ins --tasks cybermetrics:80

# Run multiple tasks
python eval/run_lighteval.py vllm \
    --model RISys-Lab/RedSage-8B-Ins \
    --tasks cybermetrics:80,mmlu:cs_security,secbench:mcq-en \
    --output-dir results/my_eval
```

## Running Evaluations

### Using the Wrapper Script (Recommended)

The `run_lighteval.py` script provides a user-friendly interface:

```bash
python eval/run_lighteval.py [backend] --model MODEL_NAME --tasks TASK_LIST [OPTIONS]
```

**Arguments:**
- `backend`: Choose `transformers` (default) or `vllm`
- `--model`: Model name or path (e.g., `RISys-Lab/RedSage-8B-Ins`)
- `--tasks`: Comma-separated list of tasks (e.g., `cybermetrics:80,mmlu:cs_security`)
- `--output-dir`: Directory to save results (default: `results`)
- `--num-fewshot`: Number of few-shot examples (default: 0)
- `--max-samples`: Limit samples per task (useful for testing)

**vLLM-specific options:**
- `--vllm-gpu-memory-utilization`: GPU memory usage (default: 0.9)
- `--vllm-max-model-len`: Maximum sequence length
- `--vllm-tensor-parallel-size`: Number of GPUs for tensor parallelism

### Using Lighteval Directly

You can also use the `lighteval` CLI directly with the `--custom-tasks` argument:

```bash
lighteval vllm "model_name=RISys-Lab/RedSage-8B-Ins" \
    --custom-tasks eval/cybersecurity_benchmarks.py \
    --tasks "lighteval|mmlu:cs_security|0"
```

## Available Tasks

The following tasks are defined in `eval/cybersecurity_benchmarks.py`.

> **Note**: For API-based endpoints or generative evaluation, use the `_em` (exact match) versions.

### CyberMetrics

Cybersecurity knowledge benchmark with different dataset sizes:

- `cybermetrics:80` - 80 questions (loglikelihood)
- `cybermetrics:500` - 500 questions (loglikelihood)
- `cybermetrics:2000` - 2,000 questions (loglikelihood)
- `cybermetrics:10000` - 10,000 questions (loglikelihood)
- `cybermetrics:80_em` - 80 questions (generative/exact match)
- `cybermetrics:500_em` - 500 questions (generative/exact match)
- `cybermetrics:2000_em` - 2,000 questions (generative/exact match)
- `cybermetrics:10000_em` - 10,000 questions (generative/exact match)

### CTI-Bench

Cyber Threat Intelligence benchmark:

- `cti_bench:cti-mcq` - Multiple choice questions (loglikelihood)
- `cti_bench:cti-mcq_ori` - MCQ with original prompt (generative, full response)
- `cti_bench:cti-mcq_em` - MCQ with exact match (generative, single line)
- `cti_bench:cti-rcm` - Root Cause Mapping (generative, single line)
- `cti_bench:cti-rcm_ori` - RCM with original prompt (generative, full response)

### MMLU Computer Security

- `mmlu:cs_security` - MMLU Computer Security subset with custom prompt

### SECURE

Security Understanding and Reasoning Evaluation:

- `secure:maet_em` - Malware Analysis and Ethical Testing
- `secure:cwet_em` - Common Weakness Enumeration Testing
- `secure:kcv_em` - Knowledge and Comprehension Validation

### SecBench

Security benchmark with multiple-choice questions:

- `secbench:mcq-en` - English MCQ (loglikelihood)
- `secbench:mcq-en_em` - English MCQ (generative/exact match)

### SecEval

Security evaluation with multi-answer questions:

- `seceval:mcqa` - Multi-choice questions with 5-shot examples
- `seceval:mcqa_0s` - Multi-choice questions with 0-shot

### RedSage MCQ

RedSage's internal cybersecurity benchmark covering multiple domains:

**Full dataset (test_10k split):**
- `redsage_mcq:cybersecurity_knowledge_generals`
- `redsage_mcq:cybersecurity_knowledge_frameworks`
- `redsage_mcq:cybersecurity_skills`
- `redsage_mcq:cybersecurity_tools`

**With context variants (includes content field):**
- `redsage_mcq_ctx:cybersecurity_knowledge_generals`
- `redsage_mcq_ctx:cybersecurity_knowledge_frameworks`
- `redsage_mcq_ctx:cybersecurity_skills`
- `redsage_mcq_ctx:cybersecurity_tools`

**Exact match variants (generative evaluation):**
- `redsage_mcq_em:cybersecurity_knowledge_generals`
- `redsage_mcq_em:cybersecurity_knowledge_frameworks`
- `redsage_mcq_em:cybersecurity_skills`
- `redsage_mcq_em:cybersecurity_tools`

**5K subset (test_5k split):**
- `redsage_mcq_5k:cybersecurity_knowledge_generals`
- `redsage_mcq_5k:cybersecurity_knowledge_frameworks`
- `redsage_mcq_5k:cybersecurity_skills`
- `redsage_mcq_5k:cybersecurity_tools_cli`
- `redsage_mcq_5k:cybersecurity_tools_kali`

And corresponding `_ctx` and `_em` variants for the 5K split.

**Note**: For `redsage_mcq` tasks, ensure you have access to the dataset `naufalso/RedSage_MCQ_Qwen_Verified` on Hugging Face.

## Advanced Usage

### Running a Complete Benchmark Suite

```bash
# Run all CyberMetrics benchmarks
python eval/run_lighteval.py vllm \
    --model RISys-Lab/RedSage-8B-DPO \
    --tasks cybermetrics:80,cybermetrics:500,cybermetrics:2000,cybermetrics:10000 \
    --output-dir results/cybermetrics_full

# Run all CTI-Bench tasks
python eval/run_lighteval.py vllm \
    --model RISys-Lab/RedSage-8B-DPO \
    --tasks cti_bench:cti-mcq,cti_bench:cti-rcm \
    --output-dir results/cti_bench_full

# Run core RedSage benchmarks
python eval/run_lighteval.py vllm \
    --model RISys-Lab/RedSage-8B-DPO \
    --tasks redsage_mcq:cybersecurity_knowledge_generals,redsage_mcq:cybersecurity_knowledge_frameworks,redsage_mcq:cybersecurity_skills,redsage_mcq:cybersecurity_tools \
    --output-dir results/redsage_full
```

### Testing with Limited Samples

```bash
# Test with only 10 samples per task
python eval/run_lighteval.py vllm \
    --model RISys-Lab/RedSage-8B-Ins \
    --tasks cybermetrics:80 \
    --max-samples 10 \
    --output-dir results/test
```

### Multi-GPU Evaluation

```bash
# Use 2 GPUs with tensor parallelism
python eval/run_lighteval.py vllm \
    --model RISys-Lab/RedSage-8B-DPO \
    --tasks cybermetrics:2000 \
    --vllm-tensor-parallel-size 2 \
    --output-dir results/multi_gpu
```

### Custom vLLM Configuration

```bash
# Configure vLLM with specific parameters
python eval/run_lighteval.py vllm \
    --model RISys-Lab/RedSage-8B-Ins \
    --tasks cybermetrics:500 \
    --vllm-gpu-memory-utilization 0.8 \
    --vllm-max-model-len 16384 \
    --output-dir results/custom_vllm
```

### Direct Lighteval Usage

For full control, use `lighteval` directly:

```bash
# Single task with vLLM
lighteval vllm "model_name=RISys-Lab/RedSage-8B-Ins,gpu_memory_utilisation=0.9" \
    --custom-tasks eval/cybersecurity_benchmarks.py \
    --tasks "lighteval|cybermetrics:80|0" \
    --output-dir results/

# Multiple tasks
lighteval vllm "model_name=RISys-Lab/RedSage-8B-Ins" \
    --custom-tasks eval/cybersecurity_benchmarks.py \
    --tasks "lighteval|mmlu:cs_security|0,lighteval|cybermetrics:80|0,lighteval|secbench:mcq-en|0" \
    --output-dir results/

# With Transformers backend
lighteval transformers "pretrained=RISys-Lab/RedSage-8B-Ins" \
    --custom-tasks eval/cybersecurity_benchmarks.py \
    --tasks "lighteval|cybermetrics:80|0" \
    --output-dir results/
```

## Troubleshooting

### Submodule Not Initialized

If you get an error about lighteval not being found:

```bash
git submodule update --init --recursive
cd eval/lighteval
pip install -e .
cd ../../
```

### Missing Dependencies

If you encounter import errors:

```bash
pip install cvss aenum
```

### CUDA Out of Memory

If you run out of GPU memory:

1. Use smaller batch size (default is 1)
2. Reduce vLLM memory utilization:
   ```bash
   python eval/run_lighteval.py vllm \
       --model RISys-Lab/RedSage-8B-Ins \
       --tasks cybermetrics:80 \
       --vllm-gpu-memory-utilization 0.7
   ```
3. Reduce max model length:
   ```bash
   python eval/run_lighteval.py vllm \
       --model RISys-Lab/RedSage-8B-Ins \
       --tasks cybermetrics:80 \
       --vllm-max-model-len 4096
   ```

### Dataset Access Issues

For RedSage MCQ tasks, ensure you have access to the dataset:

1. Log in to Hugging Face: `huggingface-cli login`
2. Request access to `naufalso/RedSage_MCQ_Qwen_Verified` if needed

### Task Name Errors

Make sure to use the correct task format:
- With wrapper script: `cybermetrics:80` (no prefix/suffix)
- With lighteval directly: `lighteval|cybermetrics:80|0` (full format)

### Performance Tips

1. **Use vLLM for faster inference**: The vLLM backend is significantly faster than Transformers
2. **Adjust batch size**: While default is 1, some tasks may benefit from larger batches
3. **Multi-GPU**: Use `--vllm-tensor-parallel-size` for very large models
4. **Test first**: Use `--max-samples 10` to test your setup before full evaluation