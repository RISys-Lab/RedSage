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

