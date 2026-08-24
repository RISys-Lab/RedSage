import argparse
import logging
from typing import List

from fineweb_security.datasets import FineWebDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download FineWeb dataset subsets.")
    parser.add_argument("config_name", type=str, nargs="*", default=["CC-MAIN-2024-18"])
    parser.add_argument("--cache_dir", type=str, default="./huggingface/hub")
    parser.add_argument("--download_remaining", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_names = args.config_name or ["CC-MAIN-2024-18"]
    fineweb = FineWebDataset(cache_dir=args.cache_dir)

    for config_name in config_names:
        logger.info("Processing configuration: %s", config_name)
        if args.download_remaining:
            remaining = fineweb.list_remaining_progress(config_name, args.output_dir)
            logger.info("Remaining parquet files for %s: %d", config_name, len(remaining))
        else:
            remaining = None

        fineweb.download_all_dataset(config_name, remaining)


if __name__ == "__main__":
    main()

