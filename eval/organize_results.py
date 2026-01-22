"""Aggregate RedSage evaluation JSON results into a single CSV.

Usage:
  python organize_results.py \
    --input results/redsage_mcq/results \
    --output results/redsage_mcq/results/redsage_mcq_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List


def _iter_json_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.json")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _extract_rows(payload: Dict[str, Any], json_path: Path) -> List[Dict[str, Any]]:
    results = payload.get("results", {}) or {}
    config_general = payload.get("config_general", {}) or {}
    model_name = config_general.get("model_name") or config_general.get("model_config", {}).get("model_name")
    if not model_name:
        model_name = str(json_path.parent)

    rows: List[Dict[str, Any]] = []
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        row: Dict[str, Any] = {
            "model": model_name,
            "task": task_name,
            "json_path": str(json_path),
        }
        for metric_name, metric_value in metrics.items():
            row[metric_name] = metric_value
        rows.append(row)
    return rows


def _pivot_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    all_columns: set[str] = set()

    for row in rows:
        model = str(row.get("model", ""))
        if model not in grouped:
            grouped[model] = {"model": model}
        target = grouped[model]

        for key, value in row.items():
            if key in {"model", "task", "json_path"}:
                continue
            column = f"{row.get('task','')}_{key}"
            target[column] = value
            all_columns.add(column)

    columns = ["model"] + sorted(all_columns)
    return [{col: grouped[model].get(col, "") for col in columns} for model in sorted(grouped)]


def _trim_model_common_prefix(rows: List[Dict[str, Any]]) -> None:
    models = [str(row.get("model", "")) for row in rows if row.get("model")]
    if len(models) < 2:
        return

    parts_list = [PurePosixPath(model).parts for model in models]
    common_parts: List[str] = []
    for parts in zip(*parts_list):
        if len(set(parts)) == 1:
            common_parts.append(parts[0])
        else:
            break

    if not common_parts:
        return

    common_prefix = "/".join(common_parts).rstrip("/") + "/"
    for row in rows:
        model = str(row.get("model", ""))
        if model.startswith(common_prefix) and len(model) > len(common_prefix):
            row["model"] = model[len(common_prefix):]


def _drop_stderr_metrics(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        keys_to_drop = [key for key in row.keys() if key.endswith("_stderr")]
        for key in keys_to_drop:
            row.pop(key, None)


def _write_csv(rows: List[Dict[str, Any]], fieldnames: List[str], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _split_fieldnames_by_metric(fieldnames: List[str]) -> Dict[str, List[str]]:
    base_cols = [col for col in ("model", "task", "json_path") if col in fieldnames]
    metrics: Dict[str, List[str]] = {}

    for col in fieldnames:
        if col in base_cols:
            continue
        metric = col.rsplit("_", 1)[-1] if "_" in col else col
        metrics.setdefault(metric, base_cols.copy()).append(col)

    return metrics


def aggregate_to_csv(
    input_dir: Path,
    output_csv: Path,
    pivot_by_model: bool = True,
    include_stderr: bool = True,
    split_metrics: bool = False,
) -> int:
    rows: List[Dict[str, Any]] = []
    for json_path in _iter_json_files(input_dir):
        try:
            payload = _load_json(json_path)
        except json.JSONDecodeError:
            continue
        rows.extend(_extract_rows(payload, json_path))

    if not rows:
        return 0

    _trim_model_common_prefix(rows)
    if not include_stderr:
        _drop_stderr_metrics(rows)

    if pivot_by_model:
        rows = _pivot_rows(rows)
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["model", "task", "json_path"]
        extra_fields = sorted({k for row in rows for k in row.keys() if k not in fieldnames})
        fieldnames.extend(extra_fields)

    if split_metrics:
        metrics = _split_fieldnames_by_metric(fieldnames)
        for metric, metric_fields in metrics.items():
            metric_output = output_csv.with_name(f"{output_csv.stem}_{metric}{output_csv.suffix}")
            _write_csv(rows, metric_fields, metric_output)
    else:
        _write_csv(rows, fieldnames, output_csv)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate JSON eval results into a single CSV.")
    parser.add_argument(
        "input",
        type=Path,
        nargs='?',
        default=Path("results/redsage_mcq"),
        help="Root directory containing JSON result files.",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs='?',
        default=Path("results/redsage_mcq_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no-pivot",
        action="store_true",
        help="Disable pivot; output in long-form rows.",
    )
    parser.add_argument(
        "--no-stderr",
        action="store_true",
        help="Exclude *_stderr metrics from output.",
    )
    parser.add_argument(
        "--split-metrics",
        action="store_true",
        help="Write one output file per metric (e.g., _acc.csv, _em.csv, _pem.csv).",
    )
    args = parser.parse_args()

    count = aggregate_to_csv(
        args.input,
        args.output,
        pivot_by_model=not args.no_pivot,
        include_stderr=not args.no_stderr,
        split_metrics=args.split_metrics,
    )
    if args.split_metrics:
        stem = args.output.stem
        suffix = args.output.suffix
        output_dir = args.output.parent
        print(
            f"Wrote {count} rows to metric-specific CSV files in {output_dir} "
            f"with pattern '{stem}_<metric>{suffix}'"
        )
    else:
        print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
