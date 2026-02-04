#!/usr/bin/env python3
"""Score OpenQA answers using an LLM-as-judge with vLLM batch inference."""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the Judge for an Open-QA cybersecurity benchmark.

## Inputs
- question: the final user-facing prompt
- reference_answer: the ground-truth answer
- model_answer: the response produced by the evaluated model

## Goal
Evaluate model_answer on two aspects: (1) factual correctness, and (2) overall answer quality (0-10).

---

## Checklist

1) Correctness
- Is the answer factually accurate?
- Does it align with the reference_answer and grounded cybersecurity knowledge?
- Any hallucinations, contradictions, or false claims?

Output rule for correctness:
- True if correct and grounded
- False if factually incorrect, hallucinated, contradicted, or clearly wrong

2) Answer Quality Score (0-10)
Rate overall quality only if at least partially correct:
- Helpfulness (does it answer the question?)
- Relevance (focused and on-topic)
- Depth (reasoning or understanding)
- Level of detail (complete and specific enough)

Scoring guide:
- 10: perfect — accurate, complete, deep, fully relevant
- 8-9: strong — minor omissions or small inaccuracies
- 6-7: moderate — useful but lacking depth or detail
- 4-5: weak — vague, shallow, or incomplete
- 1-3: poor — limited usefulness or clarity
- 0: invalid or gibberish

---

## Instructions
- Use chain-of-thought privately, but present only a final analysis in <analysis>.  
- Be strict on correctness: any factual error → correctness=False. If correctness=False, cap score at 3 or lower.  
- If correct but shallow, keep correctness=True and assign a lower score.

---

## Output Format
Return exactly these three blocks in order. Do not add text outside the tags.

<analysis>
Free-form justification. You may write anything here such as step-by-step reasoning, comparisons, errors spotted, strengths, weaknesses, etc. between the model_answer and reference_answe.
Make sure your analysis is detailed and covers all aspects of the evaluation checklist.

### Correctness
Analysis and justification for the correctness evaluation.

### Answer Quality Score
Analysis and justification for the answer quality score.

#### Helpfulness
Justification for the helpfulness aspect.

#### Relevance
Justification for the relevance aspect.

#### Depth
Justification for the depth aspect.

#### Level of Detail
Justification for the level of detail aspect.
</analysis>

<correctness>
True or False
</correctness>

<score>
0-10 (integer only)
</score>
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score OpenQA answers with vLLM judge.")
    parser.add_argument("--model", required=True, help="Judge model name or path.")
    parser.add_argument("--input", required=True, help="Input JSONL file or folder containing JSONL files.")
    parser.add_argument("--output", required=True, help="Output JSONL file or folder for scored results.")
    parser.add_argument(
        "--question-column",
        default="question",
        help="Column name for the question.",
    )
    parser.add_argument(
        "--reference-column",
        default="reference_answer",
        help="Column name for the reference answer.",
    )
    parser.add_argument(
        "--model-answer-column",
        default="generated_text",
        help="Column name for the model answer.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Disable the tokenizer chat template for formatting.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p.")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Optional max model length override for vLLM.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=3,
        help="Max retries for parsing failures.",
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


def iter_batches(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_prompt(
    question: str,
    reference_answer: str,
    model_answer: str,
    tokenizer: Optional[AutoTokenizer],
    use_chat_template: bool,
) -> str:
    user_prompt = (
        f"Question:\n```\n{question}\n```\n\n"
        f"Reference Answer:\n```\n{reference_answer}\n```\n\n"
        f"Model Answer:\n```\n{model_answer}\n```"
    )

    if use_chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


CORRECTNESS_RE = re.compile(r"<correctness>\s*(True|False)\s*</correctness>", re.DOTALL)
SCORE_RE = re.compile(r"<score>\s*(\d{1,2})\s*</score>", re.DOTALL)


def parse_judge_output(text: str) -> Optional[Tuple[bool, int]]:
    correctness_match = CORRECTNESS_RE.search(text)
    score_match = SCORE_RE.search(text)
    if not correctness_match or not score_match:
        return None

    correctness_value = correctness_match.group(1) == "True"
    score_value = int(score_match.group(1))
    if score_value < 0 or score_value > 10:
        return None
    return correctness_value, score_value


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    args = parse_args()
    logger.info("Starting OpenQA scoring with the following configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Determine if input is a file or folder
    if input_path.is_file():
        # Single file mode
        jsonl_files = [input_path]
        is_folder_mode = False
        logger.info(f"Single file mode: processing {input_path}")
    elif input_path.is_dir():
        # Folder mode
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        if not jsonl_files:
            logger.error(f"No JSONL files found in {input_path}")
            return
        is_folder_mode = True
        logger.info(f"Folder mode: found {len(jsonl_files)} JSONL file(s) to process")
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model) if not args.disable_chat_template else None
    if tokenizer:
        logger.info("Tokenizer loaded successfully")
    else:
        logger.info("Chat template disabled, tokenizer not loaded")

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
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Engine configuration: {engine_kwargs}")
    llm = LLM(model=args.model, **engine_kwargs)
    logger.info("Model loaded successfully")
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    # Process each JSONL file
    for input_file in jsonl_files:
        logger.info(f"Processing file: {input_file.name}")
        rows = load_jsonl(input_file)
        logger.info(f"  Loaded {len(rows)} rows from {input_file.name}")
        
        # Determine output path
        if is_folder_mode:
            current_output_path = output_path / input_file.name
        else:
            current_output_path = output_path

        pending_indices = list(range(len(rows)))
        trial = 0
        last_raw = {}

        while pending_indices and trial < args.max_trials:
            logger.info(f"  Trial {trial + 1}/{args.max_trials}: {len(pending_indices)} items pending")
            next_pending = []
            for batch_indices in iter_batches(pending_indices, args.batch_size):
                logger.debug(f"    Processing batch of {len(batch_indices)} items")
                prompts = []
                for idx in batch_indices:
                    row = rows[idx]
                    prompts.append(
                        build_prompt(
                            row[args.question_column],
                            row[args.reference_column],
                            row[args.model_answer_column],
                            tokenizer,
                            not args.disable_chat_template,
                        )
                    )

                results = llm.generate(prompts, sampling_params)
                parsed_count = 0
                for idx, result in zip(batch_indices, results):
                    text = result.outputs[0].text.strip()
                    last_raw[idx] = text
                    parsed = parse_judge_output(text)
                    if parsed is None:
                        next_pending.append(idx)
                        continue
                    parsed_count += 1
                    correctness, score = parsed
                    rows[idx]["judge_raw"] = text
                    rows[idx]["judge_correctness"] = correctness
                    rows[idx]["judge_score"] = score
                logger.debug(f"    Batch complete: {parsed_count}/{len(batch_indices)} items successfully parsed")

            pending_indices = next_pending
            trial += 1

        for idx in pending_indices:
            rows[idx]["judge_raw"] = last_raw.get(idx)
            rows[idx]["judge_correctness"] = None
            rows[idx]["judge_score"] = None
            rows[idx]["judge_parse_error"] = True
        
        if pending_indices:
            logger.warning(f"  {len(pending_indices)} items failed to parse after {args.max_trials} trials")

        current_output_path.parent.mkdir(parents=True, exist_ok=True)
        with current_output_path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        logger.info(f"  Saved results to {current_output_path}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting OpenQA Answer Scoring Process")
    logger.info("=" * 60)
    main()
    logger.info("=" * 60)
    logger.info("Scoring process completed")
    logger.info("=" * 60)
