#!/usr/bin/env python3
"""Generate OpenQA answers using vLLM offline batch inference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
import inspect


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenQA answers with vLLM.")
    parser.add_argument("--model", required=True, help="Model name or path.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset name (e.g. openqa).",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Dataset configuration name.",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional local JSON/JSONL file to load instead of a HF dataset.",
    )
    parser.add_argument(
        "--question-column",
        default="question",
        help="Column containing the question text.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--prompt-template",
        default="{question}",
        help="Prompt template with {question} placeholder.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt prepended to the user content.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Disable the tokenizer chat template for formatting.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p.")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Optional max model length override for vLLM.",
    )
    # vLLM Engine Arguments
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for pipeline parallelism.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use for vLLM.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per iteration.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per iteration.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights (auto, float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        help="Data type for KV cache (auto, float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        help="Format to load model weights (auto, pt, safetensors, npcache).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace Hub.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Directory to download model weights to.",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Revision of the model to use.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help="Tokenizer mode (auto, slow).",
    )
    parser.add_argument(
        "--tokenizer-revision",
        type=str,
        default=None,
        help="Revision of the tokenizer to use.",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=4,
        help="CPU swap space size in GiB per GPU.",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=0,
        help="Offload weights to CPU with this size in GiB.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager execution (no CUDA graph).",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable prefix caching for multi-turn conversations.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable chunked prefill execution.",
    )
    return parser.parse_args()


def load_data(args: argparse.Namespace, dataset_config: Optional[str]):
    LOGGER.info("Loading dataset.")
    if args.data_file:
        data_path = Path(args.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        extension = data_path.suffix.lstrip(".")
        if extension not in {"json", "jsonl"}:
            raise ValueError("Only JSON/JSONL files are supported for --data-file.")
        dataset = load_dataset("json", data_files=str(data_path), split="train")
        LOGGER.info("Loaded %d records from %s.", len(dataset), data_path)
        return dataset

    if not args.dataset:
        raise ValueError("Provide --dataset or --data-file.")

    dataset_kwargs = {"path": args.dataset}
    if dataset_config:
        dataset_kwargs["name"] = dataset_config
    dataset = load_dataset(**dataset_kwargs, split=args.split)
    if dataset_config:
        LOGGER.info(
            "Loaded %d records from %s (%s/%s).",
            len(dataset),
            args.dataset,
            dataset_config,
            args.split,
        )
    else:
        LOGGER.info("Loaded %d records from %s (%s).", len(dataset), args.dataset, args.split)
    return dataset


def iter_batches(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_prompt(
    question: str,
    prompt_template: str,
    system_prompt: Optional[str],
    tokenizer: Optional[AutoTokenizer],
    use_chat_template: bool,
) -> str:
    user_content = prompt_template.format(question=question)
    if use_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if system_prompt:
        return f"{system_prompt}\n\n{user_content}"
    return user_content


def count_lines(file_path: Path) -> int:
    """Count the number of lines in a file."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    LOGGER.info("Starting OpenQA generation.")
    if args.dataset_config:
        dataset_configs = [
            config.strip()
            for config in args.dataset_config.split(",")
            if config.strip()
        ]
    else:
        dataset_configs = [None]

    if not args.disable_chat_template:
        LOGGER.info("Loading tokenizer for chat template: %s", args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = None

    LOGGER.info("Loading vLLM model: %s", args.model)
    # Get valid EngineArgs parameters
    engine_args_sig = inspect.signature(EngineArgs.__init__)
    valid_params = set(engine_args_sig.parameters.keys()) - {'self'}
    
    # Map argument names to EngineArgs parameter names
    arg_mapping = {
        'max_model_len': 'max_model_len',
        'tensor_parallel_size': 'tensor_parallel_size',
        'pipeline_parallel_size': 'pipeline_parallel_size',
        'gpu_memory_utilization': 'gpu_memory_utilization',
        'max_num_batched_tokens': 'max_num_batched_tokens',
        'max_num_seqs': 'max_num_seqs',
        'dtype': 'dtype',
        'kv_cache_dtype': 'kv_cache_dtype',
        'load_format': 'load_format',
        'trust_remote_code': 'trust_remote_code',
        'tokenizer_mode': 'tokenizer_mode',
        'tokenizer_revision': 'tokenizer_revision',
        'model_revision': 'revision',
        'swap_space': 'swap_space',
        'cpu_offload_gb': 'cpu_offload_gb',
        'enforce_eager': 'enforce_eager',
        'enable_prefix_caching': 'enable_prefix_caching',
        'enable_chunked_prefill': 'enable_chunked_prefill',
        'download_dir': 'download_dir',
    }
    
    # Build engine arguments dynamically, only including valid parameters
    engine_kwargs = {}
    for arg_name, param_name in arg_mapping.items():
        if param_name in valid_params and hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:  # Skip None values to use EngineArgs defaults
                engine_kwargs[param_name] = value
    
    # Initialize LLM with model and engine kwargs
    llm = LLM(model=args.model, **engine_kwargs)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    base_output_path = Path(args.output_file)
    base_output_path.parent.mkdir(parents=True, exist_ok=True)

    for dataset_config in dataset_configs:
        config_label = dataset_config or "default"
        LOGGER.info("Processing dataset config: %s", config_label)
        dataset = load_data(args, dataset_config)
        records = list(dataset)
        LOGGER.info("Prepared %d records for inference (config: %s).", len(records), config_label)

        if dataset_config:
            output_path = base_output_path.with_name(
                f"{base_output_path.stem}_{dataset_config}{base_output_path.suffix}"
            )
        else:
            output_path = base_output_path

        # Check if output file already exists and has the correct number of lines
        if output_path.exists():
            existing_lines = count_lines(output_path)
            if existing_lines == len(records):
                LOGGER.info(
                    "Output file %s already exists with %d lines (matches %d records). Skipping inference (config: %s)",
                    output_path,
                    existing_lines,
                    len(records),
                    config_label,
                )
                continue
            else:
                LOGGER.warning(
                    "Output file %s exists but has %d lines (expected %d records). Removing and restarting inference (config: %s)",
                    output_path,
                    existing_lines,
                    len(records),
                    config_label,
                )
                output_path.unlink()

        LOGGER.info("Writing outputs to %s (config: %s)", output_path, config_label)

        with output_path.open("w", encoding="utf-8") as output_file:
            total_batches = (len(records) + args.batch_size - 1) // args.batch_size
            for batch_idx, batch in enumerate(iter_batches(records, args.batch_size), start=1):
                LOGGER.info(
                    "Generating batch %d/%d (config: %s)",
                    batch_idx,
                    total_batches,
                    config_label,
                )
                prompts = [
                    build_prompt(
                        item[args.question_column],
                        args.prompt_template,
                        args.system_prompt,
                        tokenizer,
                        not args.disable_chat_template,
                    )
                    for item in batch
                ]

                LOGGER.info(
                    "Running vLLM generation for %d prompts (config: %s).",
                    len(prompts),
                    config_label,
                )
                results = llm.generate(prompts, sampling_params)
                for item, result, prompt in zip(batch, results, prompts):
                    generated_text = result.outputs[0].text
                    merged = dict(item)
                    merged["prompt"] = prompt
                    merged["generated_text"] = generated_text
                    output_file.write(json.dumps(merged, ensure_ascii=False) + "\n")

    LOGGER.info("Completed OpenQA generation.")


if __name__ == "__main__":
    main()
