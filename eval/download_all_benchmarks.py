"""Utility to pre-download all benchmark datasets into the local Hugging Face cache.

Running this script fetches every dataset referenced in ``cybersecurity_benchmarks``
so subsequent evaluation runs can reuse the cached copies without downloading at
execution time. This is useful on shared clusters or offline environments where
network access is limited once jobs start.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from cybersecurity_benchmarks import TASKS_TABLE
from datasets import load_dataset


def download_all_benchmarks():
    """Download every benchmark dataset defined in ``TASKS_TABLE`` to cache.

    Iterates over all tasks, triggers a ``load_dataset`` call for each Hugging
    Face repository/subset pair, and relies on the HF datasets cache to store
    the artifacts locally. The returned datasets are not used directly; the
    side effect of caching is the goal.
    """

    for task_config in TASKS_TABLE:
        print(f"Downloading dataset for task: {task_config.name}")
        _ = load_dataset(task_config.hf_repo, task_config.hf_subset)

if __name__ == "__main__":
    download_all_benchmarks()