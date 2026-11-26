from pathlib import Path

import pytest

from raglab.documents import load_directory


def test_load_directory_reads_supported_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("Beta", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.txt").write_text("Alpha", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")

    documents = load_directory(tmp_path)

    assert [document.id for document in documents] == ["b.md", "nested/a.txt"]
    assert documents[1].text == "Alpha"
    assert documents[1].metadata == {"source": "nested/a.txt"}


def test_load_directory_accepts_custom_suffixes(tmp_path: Path) -> None:
    (tmp_path / "notes.rst").write_text("Restructured", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Markdown", encoding="utf-8")

    documents = load_directory(tmp_path, suffixes=[".RST"])

    assert [document.id for document in documents] == ["notes.rst"]


def test_load_directory_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "missing")


def test_load_directory_merges_optional_metadata_manifest(tmp_path: Path) -> None:
    import json

    (tmp_path / "runbooks").mkdir()
    (tmp_path / "runbooks" / "pg.md").write_text("Pool saturation fix.", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"runbooks/pg.md": {"kind": "runbook", "service": "postgres"}}),
        encoding="utf-8",
    )

    documents = load_directory(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata == {
        "source": "runbooks/pg.md",
        "kind": "runbook",
        "service": "postgres",
    }
