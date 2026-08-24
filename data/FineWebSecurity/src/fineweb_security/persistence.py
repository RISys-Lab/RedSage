import json
import logging
import os
import re
from json.decoder import JSONDecodeError
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", value)


def subset_output_dir(output_path: str, dataset_subset: str) -> str:
    return os.path.join(output_path, dataset_subset)


def parquet_output_dir(output_path: str, dataset_subset: str, parquet_file: str) -> str:
    parquet_stem = os.path.basename(parquet_file).replace(".parquet", "")
    return os.path.join(subset_output_dir(output_path, dataset_subset), parquet_stem)


def filtered_parquet_path(output_path: str, dataset_subset: str, parquet_idx: int) -> str:
    return os.path.join(subset_output_dir(output_path, dataset_subset), f"filtered_{parquet_idx}.parquet")


def save_relevant_document(item: Tuple[Dict[str, Any], float, str]) -> None:
    sample, probability, output_path = item
    result = {"probability": float(probability), "relevant": True}
    result.update(sample)

    os.makedirs(output_path, exist_ok=True)
    filename = sanitize_filename(str(sample["id"])) + ".json"
    file_path = os.path.join(output_path, filename)

    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4, cls=NumpyEncoder)


def load_json_documents(folder_path: str) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    if not os.path.isdir(folder_path):
        return documents

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
        except JSONDecodeError as exc:
            logger.warning("Skipping %s: invalid JSON (%s)", file_path, exc)
            continue
        except OSError as exc:
            logger.warning("Skipping %s: failed to read file (%s)", file_path, exc)
            continue

        if isinstance(document, dict):
            documents.append(document)
        else:
            logger.warning("Skipping %s: expected JSON object, got %s", file_path, type(document).__name__)

    return documents


def write_filtered_parquet(documents: Iterable[Dict[str, Any]], output_file: str) -> bool:
    document_list = list(documents)
    if not document_list:
        return False

    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    table = pa.Table.from_pylist(document_list)
    pq.write_table(table, output_file)
    return True
