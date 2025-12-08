# Evaluation

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

## Running Evaluations

You can run evaluations using the `lighteval` CLI. Use the `--custom-tasks` argument to point to our benchmark definition file.

### Basic Usage

```bash
lighteval vllm "model_name=Qwen/Qwen2.5-7B-Instruct" \
    --custom-tasks eval/cybersecurity_benchmarks.py \
    --tasks "lighteval|mmlu:cs_security|0|0"
```

### Available Tasks

The following tasks are defined in `eval/cybersecurity_benchmarks.py`:

#### CyberMetrics
- `lighteval|cybermetrics:80|0|0`
- `lighteval|cybermetrics:500|0|0`
- `lighteval|cybermetrics:2000|0|0`
- `lighteval|cybermetrics:10000|0|0`
(Add `_em` suffix for Exact Match version, e.g., `cybermetrics:80_em`)

#### CTI-Bench
- `lighteval|cti_bench:cti-mcq|0|0`
- `lighteval|cti_bench:cti-rcm|0|0`
(Variants: `_ori`, `_em`)

#### MMLU Computer Security
- `lighteval|mmlu:cs_security|0|0` (Uses custom "Answer directly" prompt)

#### SECURE
- `lighteval|secure:maet_em|0|0`
- `lighteval|secure:cwet_em|0|0`
- `lighteval|secure:kcv_em|0|0`

#### SecBench
- `lighteval|secbench:mcq-en|0|0`
(Add `_em` suffix for Exact Match version)

#### SecEval
- `lighteval|seceval:mcqa|0|0`

#### RedSage MCQ
Tasks are available for various subsets:
- `cybersecurity_knowledge_generals`
- `cybersecurity_knowledge_frameworks`
- `cybersecurity_skills`
- `cybersecurity_tools`
- `cybersecurity_tools_cli`
- `cybersecurity_tools_kali`

Format: `lighteval|redsage_mcq:{subset}|0|0`
Variants:
- `redsage_mcq:{subset}` (Default)
- `redsage_mcq_ctx:{subset}` (With Context)
- `redsage_mcq_em:{subset}` (Exact Match)

**Note**: For `redsage_mcq` tasks, make sure you have access to the dataset `naufalso/RedSage_MCQ_Qwen_Verified` on Hugging Face.

## Example: Running Multiple Tasks

```bash
lighteval vllm "model_name=Qwen/Qwen2.5-7B-Instruct" \
    --custom-tasks eval/cybersecurity_benchmarks.py \
    --tasks "lighteval|mmlu:cs_security|0|0,lighteval|cybermetrics:80|0|0" \
    --output-dir results/
```