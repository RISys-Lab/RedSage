import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Progress:
    parquet_idx: int = 0
    parquet_sample_idx: int = 0


def load_progress(progress_file: str) -> Progress:
    if not os.path.exists(progress_file):
        logger.info("Progress file not found. Starting from parquet 0, sample 0.")
        return Progress()

    try:
        with open(progress_file, "r", encoding="utf-8") as handle:
            content = json.load(handle)
    except (json.JSONDecodeError, TypeError, OSError) as exc:
        logger.warning(
            "Progress file %s is invalid (%s). Starting from parquet 0, sample 0.",
            progress_file,
            exc,
        )
        return Progress()

    if not isinstance(content, dict):
        logger.warning("Progress file %s is not a JSON object. Starting from 0.", progress_file)
        return Progress()

    return Progress(
        parquet_idx=int(content.get("parquet_idx", 0) or 0),
        parquet_sample_idx=int(content.get("parquet_sample_idx", 0) or 0),
    )


def save_progress(progress_file: str, parquet_idx: int, parquet_sample_idx: int) -> None:
    parent_dir = os.path.dirname(progress_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(progress_file, "w", encoding="utf-8") as handle:
        json.dump(
            {"parquet_idx": int(parquet_idx), "parquet_sample_idx": int(parquet_sample_idx)},
            handle,
            indent=4,
        )


def default_progress_path(output_path: str, dataset_subset: str) -> str:
    return os.path.join(output_path, f"{dataset_subset}_filter_progress.json")

