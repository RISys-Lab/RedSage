# FineWebSecurity

FineWebSecurity filters [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
with the RedSage cybersecurity BERT/ModernBERT classifier. This folder is a self-contained data-processing tool; its dependencies are optional and are not required for RedSage inference, training, or evaluation.

## Released Dataset

If you only need the filtered corpus, download
[`RISys-Lab/RedSage-CFW`](https://huggingface.co/datasets/RISys-Lab/RedSage-CFW)
from Hugging Face instead of running this pipeline locally. The dataset is gated and requires accepting the dataset access conditions on Hugging Face.

## What This Contains

- `src/fineweb_security/`: reusable package code for FineWeb access, progress
  tracking, BERT inference, output persistence, and optional Hugging Face upload.
- `src/filter_fineweb_bert_map.py`: compatibility wrapper for the production
  filtering command.
- `src/check_fineweb_progress.py`: compatibility wrapper for progress checks.
- `src/download_subset.py`: compatibility wrapper for downloading FineWeb subsets.
- `scripts/filter_fineweb_bert.sh`: tmux-based multi-GPU launcher.
- `config/fineweb_config.txt`: FineWeb subset list used by the launcher.


## Install

From this directory:

```bash
uv venv --python 3.10 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set the Hugging Face cache location if desired:

```bash
export HF_HOME="/path/to/huggingface/cache"
```

Set a token for private model or dataset access:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Run Filtering

Preferred module command:

```bash
PYTHONPATH=src uv run python -m fineweb_security.cli.filter_bert \
  --dataset_subset CC-MAIN-2024-18 \
  --batch_size 640 \
  --parallel_worker 2 \
  --max_length 1024 \
  --threshold 0.875 \
  --output_path outputs/ \
  --hf_token "$HF_TOKEN"
```

Compatibility wrapper:

```bash
uv run python src/filter_fineweb_bert_map.py \
  --dataset_subset CC-MAIN-2024-18 \
  --batch_size 640 \
  --parallel_worker 2 \
  --hf_token "$HF_TOKEN"
```

For year-based tmux scheduling across available GPUs:

```bash
./scripts/filter_fineweb_bert.sh 2024
```

The launcher reads `config/fineweb_config.txt`, filters subsets containing the
requested year, and runs one filtering command per matching subset.

## Output Layout

Generated outputs preserve the existing layout:

```text
outputs/
  CC-MAIN-2024-18_filter_progress.json
  CC-MAIN-2024-18/
    000_00000/
      <document-id>.json
    filtered_0.parquet
```

Each relevant document JSON includes the original FineWeb fields plus:

- `probability`: classifier positive-class probability.
- `relevant`: always `true` for saved documents.
- `idx`: original index within the processed parquet file.

Each completed parquet shard is aggregated to
`outputs/<subset>/filtered_<parquet_idx>.parquet`.

## Resume And Remaining Work

Progress is stored as JSON:

```json
{
  "parquet_idx": 0,
  "parquet_sample_idx": 0
}
```

By default the progress file is
`outputs/<subset>_filter_progress.json`. Override it with `--progress_path`.

Check progress:

```bash
uv run python src/check_fineweb_progress.py \
  --dataset_subset CC-MAIN-2024-18 \
  --output_path outputs/
```

Process only parquet files that do not yet have non-empty output folders:

```bash
uv run python src/filter_fineweb_bert_map.py \
  --dataset_subset CC-MAIN-2024-18 \
  --complete_remaining \
  --hf_token "$HF_TOKEN"
```

## Hugging Face Upload

Filtering writes local parquet files by default. To upload each filtered parquet
after it is produced:

```bash
uv run python src/filter_fineweb_bert_map.py \
  --dataset_subset CC-MAIN-2024-18 \
  --upload_to_hub \
  --hf_repo RISys-Lab/fineweb_cybersecurity \
  --hf_token "$HF_TOKEN"
```

The command creates the dataset repo and a branch named after the dataset subset when needed.

## Download FineWeb Subsets

```bash
uv run python src/download_subset.py CC-MAIN-2024-18 --cache_dir ./huggingface/hub
```

Download only files that are not represented by existing outputs:

```bash
uv run python src/download_subset.py CC-MAIN-2024-18 \
  --download_remaining \
  --output_dir outputs/
```

## Docker

Build from this directory:

```bash
docker build -t fineweb-security .
```

Run with GPU access:

```bash
docker run --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /path/to/outputs:/app/outputs \
  -v /path/to/huggingface_cache:/app/huggingface \
  fineweb-security
```

Override the default year:

```bash
docker run --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /path/to/outputs:/app/outputs \
  fineweb-security /app/scripts/filter_fineweb_bert.sh 2024
```
