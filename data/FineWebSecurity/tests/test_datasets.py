from fineweb_security.datasets import FineWebDataset


def test_list_remaining_progress(tmp_path):
    fineweb = FineWebDataset()
    fineweb.get_parquet_list = lambda config_name: [
        "data/CC-MAIN-2024-18/000_00000.parquet",
        "data/CC-MAIN-2024-18/000_00001.parquet",
    ]

    completed = tmp_path / "CC-MAIN-2024-18" / "000_00000"
    completed.mkdir(parents=True)
    (completed / "doc.json").write_text("{}", encoding="utf-8")

    assert fineweb.list_remaining_progress("CC-MAIN-2024-18", str(tmp_path)) == [
        (1, "000_00001")
    ]


def test_iterate_and_continue_non_streaming_uses_select(monkeypatch):
    class FakeDataset:
        def __init__(self):
            self.selected = None

        def __len__(self):
            return 10

        def select(self, indices):
            self.selected = list(indices)
            return self

        def skip(self, _):
            raise AssertionError("skip() should not be called for non-streaming datasets")

    fake_dataset = FakeDataset()

    def fake_load_dataset(*args, **kwargs):
        return fake_dataset

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    fineweb = FineWebDataset()
    fineweb.get_parquet_list = lambda _: ["data/CC-MAIN-2024-18/000_00000.parquet"]

    items = list(
        fineweb.iterate_and_continue(
            "CC-MAIN-2024-18",
            start_parquet_idx=0,
            start_sample_idx=3,
            streaming=False,
        )
    )
    assert len(items) == 1
    assert fake_dataset.selected == list(range(3, 10))
