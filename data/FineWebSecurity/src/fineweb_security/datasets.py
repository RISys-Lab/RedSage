import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from huggingface_hub import HfFileSystem, get_token

logger = logging.getLogger(__name__)


class FineWebDataset:
    def __init__(self, repo_id: str = "HuggingFaceFW/fineweb", cache_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.fs = HfFileSystem()
        self.token = get_token()

    def get_parquet_list(self, config_name: str) -> List[str]:
        prefix = f"datasets/{self.repo_id}/data/{config_name}"
        parquet_files = self.fs.ls(prefix, detail=False)
        return sorted(
            file.replace(f"datasets/{self.repo_id}/", "")
            for file in parquet_files
            if file.endswith(".parquet")
        )

    def get_config_metadata(self, config_name: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        dataset_name = self.repo_id.replace("/", "%2F")
        response = requests.get(
            f"https://datasets-server.huggingface.co/size?dataset={dataset_name}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()

        for config in response.json().get("size", {}).get("configs", []):
            if config.get("config") == config_name:
                return config

        raise ValueError(f"Config '{config_name}' not found in {self.repo_id}.")

    def iterate_and_continue(
        self,
        config_name: str,
        start_parquet_idx: int = 0,
        start_sample_idx: int = 0,
        streaming: bool = False,
    ) -> Iterator[Tuple[Any, str, int]]:
        parquet_files = self.get_parquet_list(config_name)
        logger.info("Found %d parquet files for %s.", len(parquet_files), config_name)

        if start_parquet_idx >= len(parquet_files):
            logger.warning(
                "Start parquet index %d exceeds available file count %d.",
                start_parquet_idx,
                len(parquet_files),
            )
            return

        for offset, parquet_file in enumerate(parquet_files[start_parquet_idx:]):
            parquet_idx = start_parquet_idx + offset
            try:
                from datasets import load_dataset

                dataset = load_dataset(
                    self.repo_id,
                    config_name,
                    split="train",
                    data_files=parquet_file,
                    streaming=streaming,
                    cache_dir=self.cache_dir,
                )
                if offset == 0 and start_sample_idx > 0:
                    logger.info("Skipping %d examples in %s.", start_sample_idx, parquet_file)
                    dataset = dataset.skip(start_sample_idx)

                yield dataset, parquet_file, parquet_idx
            except Exception:
                logger.exception("Error loading parquet file %s.", parquet_file)
                continue

    def list_remaining_progress(self, config_name: str, output_dir: str) -> List[Tuple[int, str]]:
        parquet_files = [
            os.path.splitext(os.path.basename(path))[0]
            for path in self.get_parquet_list(config_name)
        ]
        subset_dir = os.path.join(output_dir, config_name)

        if not os.path.isdir(subset_dir):
            logger.info("Output directory does not exist yet: %s.", subset_dir)
            completed_folders: List[str] = []
        else:
            completed_folders = [
                folder
                for folder in os.listdir(subset_dir)
                if os.path.isdir(os.path.join(subset_dir, folder))
                and os.listdir(os.path.join(subset_dir, folder))
            ]

        return [
            (parquet_idx, parquet_name)
            for parquet_idx, parquet_name in enumerate(parquet_files)
            if parquet_name not in completed_folders
        ]

    def iterate_remaining(
        self,
        config_name: str,
        output_dir: str,
        streaming: bool = False,
    ) -> Iterator[Tuple[Any, str, int]]:
        remaining = self.list_remaining_progress(config_name, output_dir)
        if not remaining:
            logger.warning("No remaining parquet files to process.")
            return

        parquet_paths = self.get_parquet_list(config_name)
        parquet_path_map = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in parquet_paths
        }

        for parquet_idx, parquet_name in remaining:
            parquet_path = parquet_path_map.get(parquet_name)
            if parquet_path is None:
                logger.error("Could not resolve parquet file identifier %s.", parquet_name)
                continue

            try:
                from datasets import load_dataset

                dataset = load_dataset(
                    self.repo_id,
                    config_name,
                    split="train",
                    data_files=parquet_path,
                    streaming=streaming,
                    cache_dir=self.cache_dir,
                )
                yield dataset, parquet_path, parquet_idx
            except Exception:
                logger.exception("Error loading parquet file %s.", parquet_path)
                continue

    def download_all_dataset(
        self,
        config_name: str,
        remaining_parquet_files: Optional[List[Tuple[int, str]]] = None,
    ) -> None:
        parquet_files = self.get_parquet_list(config_name)
        remaining_names = None
        if remaining_parquet_files is not None:
            remaining_names = {name for _, name in remaining_parquet_files}

        for parquet_file in parquet_files:
            parquet_name = os.path.splitext(os.path.basename(parquet_file))[0]
            if remaining_names is not None and parquet_name not in remaining_names:
                logger.info("Skipping already completed file: %s.", parquet_file)
                continue

            from datasets import load_dataset

            dataset = load_dataset(
                self.repo_id,
                config_name,
                split="train",
                data_files=parquet_file,
                cache_dir=self.cache_dir,
            )
            logger.info("Downloaded %s to %s: %s", parquet_file, self.cache_dir, dataset)
