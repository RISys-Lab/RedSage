import numpy as np
import sys
import types

import fineweb_security.cli.filter_bert as filter_bert
from fineweb_security.progress import Progress


class RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item, block=True, timeout=None):
        self.items.append(item)


def test_process_batch_filters_by_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(filter_bert, "predict_batch", lambda token_data, model: np.array([0.2, 0.93]))
    saved_progress = []
    monkeypatch.setattr(
        filter_bert,
        "save_progress",
        lambda progress_file, parquet_idx, parquet_sample_idx: saved_progress.append(
            (progress_file, parquet_idx, parquet_sample_idx)
        ),
    )

    queue = RecordingQueue()
    batch = {
        "input_ids": [[1], [2]],
        "attention_mask": [[1], [1]],
        "id": ["low", "high"],
        "text": ["general text", "security text"],
        "dump": ["dump", "dump"],
    }

    filter_bert.process_batch(
        batch,
        models=[object()],
        batch_indices=[10, 11],
        rank_idx=0,
        threshold=0.875,
        output_path=str(tmp_path),
        save_queue=queue,
        save_frequency=1,
        progress_file=str(tmp_path / "progress.json"),
        parquet_idx=4,
    )

    assert len(queue.items) == 1
    sample, probability, output_path = queue.items[0]
    assert sample["id"] == "high"
    assert sample["idx"] == 11
    assert probability == 0.93
    assert output_path == str(tmp_path)
    assert saved_progress[-1] == (str(tmp_path / "progress.json"), 4, 12)


def test_main_uses_start_idx_when_progress_file_missing(monkeypatch, tmp_path):
    captured = {}

    class FakeQueue:
        def put(self, *_args, **_kwargs):
            return None

    class FakeManager:
        def Queue(self, maxsize):
            captured["queue_size"] = maxsize
            return FakeQueue()

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def join(self):
            return None

    class FakeDataset:
        def get_parquet_list(self, _subset):
            return []

        def iterate_and_continue(self, _subset, start_parquet_idx, start_sample_idx, streaming):
            captured["start_parquet_idx"] = start_parquet_idx
            captured["start_sample_idx"] = start_sample_idx
            captured["streaming"] = streaming
            return iter([])

        def iterate_remaining(self, *_args, **_kwargs):
            return iter([])

    monkeypatch.setattr(filter_bert, "Manager", lambda: FakeManager())
    monkeypatch.setattr(filter_bert.threading, "Thread", FakeThread)
    monkeypatch.setattr(filter_bert, "save_worker", lambda save_queue: None)
    monkeypatch.setattr(filter_bert, "load_model", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(filter_bert, "warmup_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(filter_bert, "FineWebDataset", lambda cache_dir: FakeDataset())
    monkeypatch.setattr(filter_bert, "load_progress", lambda _path: Progress())
    monkeypatch.setattr(filter_bert, "save_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(filter_bert.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(filter_bert, "subset_output_dir", lambda output_path, dataset_subset: str(tmp_path))
    monkeypatch.setattr(filter_bert.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "prefetch_generator",
        types.SimpleNamespace(BackgroundGenerator=lambda iterator: iterator),
    )

    filter_bert.main(
        [
            "--dataset_subset",
            "CC-MAIN-2024-18",
            "--output_path",
            str(tmp_path),
            "--progress_path",
            str(tmp_path / "progress.json"),
            "--start_idx",
            "7",
            "--parallel_worker",
            "1",
        ]
    )

    assert captured["start_parquet_idx"] == 7
    assert captured["start_sample_idx"] == 0
    assert captured["streaming"] is False
