#!/usr/bin/env python3
"""Score OpenQA answers using an LLM-as-judge with vLLM batch inference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
    parser.add_argument("--input-file", required=True, help="Input JSONL file.")
    parser.add_argument("--output-file", required=True, help="Output JSONL file.")
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
        "--use-chat-template",
        action="store_true",
        help="Apply the tokenizer chat template for formatting.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens.")
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
    parser.add_argument(
        "--max-trials",
        type=int,
        default=3,
        help="Max retries for parsing failures.",
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
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    rows = load_jsonl(input_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model) if args.use_chat_template else None

    llm = LLM(model=args.model, max_model_len=args.max_model_len)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    pending_indices = list(range(len(rows)))
    trial = 0
    last_raw = {}

    while pending_indices and trial < args.max_trials:
        next_pending = []
        for batch_indices in iter_batches(pending_indices, args.batch_size):
            prompts = []
            for idx in batch_indices:
                row = rows[idx]
                prompts.append(
                    build_prompt(
                        row[args.question_column],
                        row[args.reference_column],
                        row[args.model_answer_column],
                        tokenizer,
                        args.use_chat_template,
                    )
                )

            results = llm.generate(prompts, sampling_params)
            for idx, result in zip(batch_indices, results):
                text = result.outputs[0].text.strip()
                last_raw[idx] = text
                parsed = parse_judge_output(text)
                if parsed is None:
                    next_pending.append(idx)
                    continue
                correctness, score = parsed
                rows[idx]["judge_raw"] = text
                rows[idx]["judge_correctness"] = correctness
                rows[idx]["judge_score"] = score

        pending_indices = next_pending
        trial += 1

    for idx in pending_indices:
        rows[idx]["judge_raw"] = last_raw.get(idx)
        rows[idx]["judge_correctness"] = None
        rows[idx]["judge_score"] = None
        rows[idx]["judge_parse_error"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
