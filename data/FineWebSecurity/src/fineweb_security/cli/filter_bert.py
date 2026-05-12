import argparse
import datetime
import logging
import os
import threading
import time
from json.decoder import JSONDecodeError
from queue import Empty, Full
from typing import Any, Dict, List

import numpy as np
import torch
from multiprocess import Manager, set_start_method
from tqdm import tqdm

from fineweb_security.bert import load_model, predict_batch, warmup_model
from fineweb_security.datasets import FineWebDataset
from fineweb_security.hub import ensure_dataset_branch, resolve_token, upload_parquet
from fineweb_security.persistence import (
    filtered_parquet_path,
    load_json_documents,
    parquet_output_dir,
    save_relevant_document,
    subset_output_dir,
    write_filtered_parquet,
)
from fineweb_security.progress import default_progress_path, load_progress, save_progress

os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"

logger = logging.getLogger(__name__)


def setup_logging(console_level: int = logging.INFO, log_file: str | None = None) -> None:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def save_worker(save_queue: Any) -> None:
    logger.info("Save worker thread started.")
    saved_count = 0
    while True:
        try:
            item = save_queue.get(timeout=5)
            if item is None:
                break

            save_relevant_document(item)
            saved_count += 1
            if saved_count % 1000 == 0:
                logger.info("Save worker saved %d documents.", saved_count)
        except Empty:
            continue
        except Exception:
            logger.exception("Exception in save worker.")
            time.sleep(1)

    logger.info("Save worker finished after saving %d documents.", saved_count)


def process_batch(
    batch_data: Dict[str, List[Any]],
    models: List[torch.nn.Module],
    batch_indices: List[int],
    rank_idx: int | None,
    threshold: float,
    output_path: str,
    save_queue: Any,
    save_frequency: int,
    progress_file: str,
    parquet_idx: int,
) -> None:
    token_data = {
        "input_ids": batch_data.pop("input_ids"),
        "attention_mask": batch_data.pop("attention_mask"),
    }
    model_idx = (rank_idx or 0) % len(models)
    probabilities = predict_batch(token_data, models[model_idx])

    relevant_indices = np.where(probabilities >= threshold)[0]
    for idx_in_batch in relevant_indices:
        sample = {key: values[idx_in_batch] for key, values in batch_data.items()}
        sample["idx"] = batch_indices[idx_in_batch]
        item_to_save = (sample, probabilities[idx_in_batch], output_path)
        try:
            save_queue.put(item_to_save, block=True, timeout=60)
        except Full:
            logger.error("Save queue is full. Dropping item %s.", sample.get("id", "UNKNOWN_ID"))
        except Exception:
            logger.exception("Failed to put item on save queue.")

    last_processed_index = batch_indices[-1]
    current_batch_number = last_processed_index // max(len(batch_indices), 1)
    if current_batch_number % save_frequency == 0:
        save_progress(progress_file, parquet_idx, last_processed_index)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_subset", type=str, default="CC-MAIN-2024-18")
    parser.add_argument("--dataset_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=640)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--progress_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="outputs/")
    parser.add_argument(
        "--model_name",
        type=str,
        default="RISys-Lab/CyberSec-Text-Classification-ModernBert-Base",
    )
    parser.add_argument("--threshold", type=float, default=0.875)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--cache_dir", type=str, default="./huggingface/hub")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--save_frequency", type=int, default=10)
    parser.add_argument("--save_queue_size", type=int, default=10000)
    parser.add_argument("--num_proc_tokenize", type=int, default=max((os.cpu_count() or 4) // 4, 1))
    parser.add_argument("--parallel_worker", type=int, default=3)
    parser.add_argument("--hf_repo", type=str, default="RISys-Lab/fineweb_cybersecurity")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--upload_to_hub", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--complete_remaining", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter FineWeb with the cybersecurity BERT/ModernBERT classifier."
    )
    _add_arguments(parser)
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    log_file_path = args.log_file
    if log_file_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join("logs", f"filter_{args.dataset_subset}_{timestamp}.log")

    setup_logging(getattr(logging, args.log_level.upper()), log_file_path)

    hf_token = None
    if args.upload_to_hub:
        hf_token = resolve_token(args.hf_token)
        ensure_dataset_branch(args.hf_repo, args.dataset_subset, hf_token)

    if args.max_length > 512 and "bert" in args.model_name.lower():
        logger.warning(
            "max_length is %d, but standard BERT models usually limit to 512. "
            "Ensure %s supports this length.",
            args.max_length,
            args.model_name,
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Switching to CPU.")
        args.device = "cpu"
    elif args.device == "cuda":
        logger.info("Using CUDA device: %s.", torch.cuda.get_device_name(0))

    os.makedirs(subset_output_dir(args.output_path, args.dataset_subset), exist_ok=True)
    progress_file = args.progress_path or default_progress_path(args.output_path, args.dataset_subset)
    progress = load_progress(progress_file)

    manager = Manager()
    save_queue = manager.Queue(maxsize=args.save_queue_size)
    save_thread = threading.Thread(
        target=save_worker,
        kwargs={"save_queue": save_queue},
        name="SaveWorkerThread",
        daemon=False,
    )
    save_thread.start()

    models = []
    tokenizer = None
    for worker_idx in range(args.parallel_worker):
        model, tokenizer = load_model(args.model_name, args.device, args.cache_dir, args.compile_model)
        warmup_model(model, tokenizer, args.batch_size, args.max_length, args.device)
        logger.info("Model %d warmup complete.", worker_idx + 1)
        models.append(model)

    fineweb = FineWebDataset(cache_dir=args.cache_dir)
    total_parquet_files = len(fineweb.get_parquet_list(args.dataset_subset))

    if args.complete_remaining:
        dataset_iterator = fineweb.iterate_remaining(args.dataset_subset, args.output_path, streaming=False)
    else:
        dataset_iterator = fineweb.iterate_and_continue(
            args.dataset_subset,
            start_parquet_idx=progress.parquet_idx,
            start_sample_idx=progress.parquet_sample_idx,
            streaming=False,
        )

    try:
        try:
            from prefetch_generator import BackgroundGenerator

            dataset_iterator = BackgroundGenerator(dataset_iterator)
        except ImportError:
            logger.warning("prefetch_generator not available. Using standard iteration.")

        processed_parquet_count = 0
        last_pq_idx = progress.parquet_idx
        for fineweb_dataset, parquet_file, parquet_idx in dataset_iterator:
            last_pq_idx = parquet_idx
            logger.info(
                "Processing parquet file %s (%d / %d).",
                parquet_file,
                parquet_idx,
                total_parquet_files,
            )

            current_output_dir = parquet_output_dir(args.output_path, args.dataset_subset, parquet_file)
            os.makedirs(current_output_dir, exist_ok=True)

            logger.info("Tokenizing %s.", parquet_file)
            tokenized_dataset = fineweb_dataset.map(
                lambda batch: tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=args.max_length,
                    padding="max_length",
                ),
                batched=True,
                batch_size=args.batch_size,
                desc=f"Tokenizing {parquet_file} ({parquet_idx} / {total_parquet_files})",
                num_proc=args.num_proc_tokenize,
            )

            logger.info("Running inference for %s.", parquet_file)
            try:
                tokenized_dataset.map(
                    lambda batch, batch_indices, rank_idx: process_batch(
                        batch,
                        models,
                        batch_indices,
                        rank_idx,
                        args.threshold,
                        current_output_dir,
                        save_queue,
                        args.save_frequency,
                        progress_file,
                        parquet_idx,
                    ),
                    batched=True,
                    batch_size=args.batch_size,
                    with_indices=True,
                    with_rank=True,
                    desc=f"Processing {parquet_file}",
                    num_proc=args.parallel_worker,
                )
            except Exception:
                logger.exception("Error during processing of %s. Skipping to next file.", parquet_file)
                save_progress(progress_file, parquet_idx + 1, 0)
                continue

            documents = load_json_documents(current_output_dir)
            if not documents:
                logger.warning("No relevant documents found in %s.", parquet_file)
                save_progress(progress_file, parquet_idx + 1, 0)
                continue

            output_parquet = filtered_parquet_path(args.output_path, args.dataset_subset, parquet_idx)
            try:
                write_filtered_parquet(documents, output_parquet)
            except (JSONDecodeError, OSError):
                logger.exception("Failed to write filtered parquet for %s.", parquet_file)
                continue

            logger.info("Filtered dataset saved to %s.", output_parquet)
            if args.upload_to_hub and hf_token:
                upload_parquet(
                    output_parquet,
                    parquet_file,
                    args.hf_repo,
                    args.dataset_subset,
                    hf_token,
                    parquet_idx,
                )
                logger.info("Uploaded filtered dataset for parquet %d.", parquet_idx)

            processed_parquet_count += 1
            save_progress(progress_file, parquet_idx + 1, 0)

        save_progress(progress_file, last_pq_idx + 1, 0)
        logger.info("Processing complete. Processed %d parquet files.", processed_parquet_count)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Progress has been saved periodically.")
    finally:
        logger.info("Waiting for save worker to finish.")
        save_queue.put(None)
        save_thread.join()


def entrypoint() -> None:
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()


if __name__ == "__main__":
    entrypoint()
