import argparse
import logging
import os
from typing import List

from fineweb_security.datasets import FineWebDataset
from fineweb_security.progress import default_progress_path, load_progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_progress_folder(subset_name: str, output_path: str) -> None:
    subset_path = os.path.join(output_path, subset_name)
    if not os.path.isdir(subset_path):
        print(f"- Subset folder {subset_path} does not exist.")
        return

    progress_parquet = sorted(f for f in os.listdir(subset_path) if f.endswith(".parquet"))
    progress_subfolders = sorted(
        f for f in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, f))
    )
    print(
        f"- {subset_name}: {len(progress_parquet)} parquet files, "
        f"{len(progress_subfolders)} JSON output folders."
    )
    for subfolder in progress_subfolders:
        subfolder_path = os.path.join(subset_path, subfolder)
        json_count = len([f for f in os.listdir(subfolder_path) if f.endswith(".json")])
        print(f"- Found {json_count} JSON files in {subfolder_path}.")


def check_subset_progress(
    subset_name: str,
    output_path: str,
    progress_filename_pattern: str,
    fineweb: FineWebDataset,
    debug: bool = False,
) -> None:
    progress_filename = progress_filename_pattern.format(subset=subset_name)
    progress_file = os.path.join(output_path, progress_filename)
    if progress_filename_pattern == "{subset}_filter_progress.json":
        progress_file = default_progress_path(output_path, subset_name)

    progress = load_progress(progress_file)
    try:
        total_parquet_files = len(fineweb.get_parquet_list(subset_name))
    except Exception as exc:
        logger.error("Failed to get parquet list for %s: %s", subset_name, exc)
        return

    percentage = (
        100.0
        if progress.parquet_idx >= total_parquet_files and total_parquet_files > 0
        else (progress.parquet_idx / total_parquet_files) * 100 if total_parquet_files else 0.0
    )

    print("\n--- FineWebSecurity Filtering Progress ---")
    print(f"Dataset Subset: {subset_name}")
    print(f"Progress File:  {progress_file}")
    print(f"Total Parquet Files: {total_parquet_files}")
    print(f"Current Parquet Index: {progress.parquet_idx} (out of {total_parquet_files})")
    print(f"Current Sample Index: {progress.parquet_sample_idx}")
    print(f"Progress: {percentage:.2f}%")
    if debug:
        check_progress_folder(subset_name, output_path)
    print("------------------------------------------\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check FineWebSecurity filtering progress.")
    parser.add_argument("--dataset_subset", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="outputs/")
    parser.add_argument("--progress_filename_pattern", type=str, default="{subset}_filter_progress.json")
    parser.add_argument("--debug", action="store_true")
    return parser


def _discover_subsets(output_path: str) -> List[str]:
    if not os.path.isdir(output_path):
        return []
    subsets = []
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path) and item.startswith("CC-MAIN-"):
            subsets.append(item)
        elif item.endswith("_filter_progress.json"):
            subsets.append(item.replace("_filter_progress.json", ""))
    return sorted(set(subsets))


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    fineweb = FineWebDataset()

    if args.dataset_subset:
        check_subset_progress(
            args.dataset_subset,
            args.output_path,
            args.progress_filename_pattern,
            fineweb,
            args.debug,
        )
        return

    subsets = _discover_subsets(args.output_path)
    if not subsets:
        logger.warning("No dataset subsets found in %s.", args.output_path)
        return

    for subset in subsets:
        check_subset_progress(subset, args.output_path, args.progress_filename_pattern, fineweb, args.debug)


if __name__ == "__main__":
    main()

