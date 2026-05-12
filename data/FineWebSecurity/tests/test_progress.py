import json

from fineweb_security.progress import Progress, default_progress_path, load_progress, save_progress


def test_progress_round_trip(tmp_path):
    progress_file = tmp_path / "progress.json"

    save_progress(str(progress_file), parquet_idx=3, parquet_sample_idx=42)

    assert load_progress(str(progress_file)) == Progress(parquet_idx=3, parquet_sample_idx=42)


def test_load_progress_missing_and_corrupt_file(tmp_path):
    assert load_progress(str(tmp_path / "missing.json")) == Progress()

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{not-json", encoding="utf-8")

    assert load_progress(str(corrupt_file)) == Progress()


def test_default_progress_path():
    assert default_progress_path("outputs", "CC-MAIN-2024-18") == (
        "outputs/CC-MAIN-2024-18_filter_progress.json"
    )

