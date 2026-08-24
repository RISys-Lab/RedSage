import json

from fineweb_security.persistence import (
    filtered_parquet_path,
    load_json_documents,
    parquet_output_dir,
    sanitize_filename,
    save_relevant_document,
    subset_output_dir,
    write_filtered_parquet,
)


def test_output_paths_and_filename_sanitization():
    assert sanitize_filename("abc/def:ghi") == "abc_def_ghi"
    assert subset_output_dir("outputs", "CC-MAIN-2024-18") == "outputs/CC-MAIN-2024-18"
    assert parquet_output_dir("outputs", "CC-MAIN-2024-18", "data/000_00000.parquet") == (
        "outputs/CC-MAIN-2024-18/000_00000"
    )
    assert filtered_parquet_path("outputs", "CC-MAIN-2024-18", 2) == (
        "outputs/CC-MAIN-2024-18/filtered_2.parquet"
    )


def test_save_load_and_write_filtered_parquet(tmp_path):
    output_dir = tmp_path / "json"
    sample = {"id": "doc/1", "text": "security text", "dump": "CC-MAIN"}

    save_relevant_document((sample, 0.91, str(output_dir)))

    saved_file = output_dir / "doc_1.json"
    assert saved_file.exists()

    saved = json.loads(saved_file.read_text(encoding="utf-8"))
    assert saved["probability"] == 0.91
    assert saved["relevant"] is True
    assert saved["text"] == "security text"

    documents = load_json_documents(str(output_dir))
    assert len(documents) == 1

    parquet_file = tmp_path / "filtered.parquet"
    assert write_filtered_parquet(documents, str(parquet_file)) is True
    assert parquet_file.exists()


def test_write_filtered_parquet_skips_empty(tmp_path):
    assert write_filtered_parquet([], str(tmp_path / "empty.parquet")) is False

