import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    device: str,
    cache_dir: str,
    compile_model: bool = False,
) -> Tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading model %s on %s.", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    if compile_model and hasattr(torch, "compile") and device == "cuda":
        try:
            logger.info("Compiling model with torch.compile().")
            model = torch.compile(model)
        except Exception as exc:
            logger.warning("torch.compile failed (%s). Continuing without compilation.", exc)

    return model, tokenizer


def predict_batch(token_data: Dict[str, List[int]], model: torch.nn.Module) -> np.ndarray:
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor(token_data["input_ids"], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(token_data["attention_mask"], dtype=torch.long, device=device),
    }

    with torch.no_grad(), torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]

    return probs.cpu().float().numpy()


def warmup_model(model: torch.nn.Module, tokenizer: Any, batch_size: int, max_length: int, device: str) -> None:
    warmup_batch_size = max(1, min(batch_size, 8))
    warmup_max_length = max(1, min(max_length, 32))
    dummy_input_ids = torch.randint(
        0,
        tokenizer.vocab_size,
        (warmup_batch_size, warmup_max_length),
    )
    dummy_attention_mask = torch.ones((warmup_batch_size, warmup_max_length))
    predict_batch(
        {
            "input_ids": dummy_input_ids.tolist(),
            "attention_mask": dummy_attention_mask.tolist(),
        },
        model,
    )
    if device == "cuda":
        torch.cuda.synchronize()


def tokenize_batch(tokenizer: Any, texts: List[str], max_length: int) -> Dict[str, List[int]]:
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
