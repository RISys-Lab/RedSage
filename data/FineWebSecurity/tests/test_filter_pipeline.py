import numpy as np

import fineweb_security.cli.filter_bert as filter_bert


class RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item, block=True, timeout=None):
        self.items.append(item)


def test_process_batch_filters_by_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(filter_bert, "predict_batch", lambda token_data, model: np.array([0.2, 0.93]))

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
        save_frequency=10,
        progress_file=str(tmp_path / "progress.json"),
        parquet_idx=4,
    )

    assert len(queue.items) == 1
    sample, probability, output_path = queue.items[0]
    assert sample["id"] == "high"
    assert sample["idx"] == 11
    assert probability == 0.93
    assert output_path == str(tmp_path)

