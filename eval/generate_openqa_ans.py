#!/usr/bin/env python3
"""Generate OpenQA answers using vLLM offline batch inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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
    parser.add_argument("--split", default="train", help="Dataset split.")
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
        "--use-chat-template",
        action="store_true",
        help="Apply the tokenizer chat template for formatting.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p.")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max model length override for vLLM.",
    )
    return parser.parse_args()


def load_data(args: argparse.Namespace):
    if args.data_file:
        data_path = Path(args.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        extension = data_path.suffix.lstrip(".")
        if extension not in {"json", "jsonl"}:
            raise ValueError("Only JSON/JSONL files are supported for --data-file.")
        return load_dataset("json", data_files=str(data_path), split="train")

    if not args.dataset:
        raise ValueError("Provide --dataset or --data-file.")

    dataset_kwargs = {"path": args.dataset}
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config
    return load_dataset(**dataset_kwargs, split=args.split)


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


def main() -> None:
    args = parse_args()
    dataset = load_data(args)
    records = list(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model) if args.use_chat_template else None

    llm = LLM(model=args.model, max_model_len=args.max_model_len)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for batch in iter_batches(records, args.batch_size):
            prompts = [
                build_prompt(
                    item[args.question_column],
                    args.prompt_template,
                    args.system_prompt,
                    tokenizer,
                    args.use_chat_template,
                )
                for item in batch
            ]

            results = llm.generate(prompts, sampling_params)
            for item, result in zip(batch, results):
                generated_text = result.outputs[0].text
                merged = dict(item)
                merged["generated_text"] = generated_text
                output_file.write(json.dumps(merged, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
