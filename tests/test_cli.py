from pathlib import Path

import pytest

from raglab import __version__
from raglab.cli import main


def test_ask_prints_answer_and_sources(
    examples_data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["ask", "How long do I have to request a refund?", "--data", str(examples_data)])

    output = capsys.readouterr().out
    assert code == 0
    assert "Answer:" in output
    assert "30 days" in output
    assert "[1] refunds.md" in output


def test_ask_can_show_the_prompt(examples_data: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "ask",
            "Where is my data stored?",
            "--data",
            str(examples_data),
            "--retriever",
            "bm25",
            "--show-prompt",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert output.startswith("Prompt:")
    assert "Question: Where is my data stored?" in output


def test_search_lists_ranked_chunks(
    examples_data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["search", "keyboard shortcut new note", "--data", str(examples_data), "--k", "2"])

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines[0].startswith("[1] keyboard_shortcuts.md#")
    assert len([line for line in lines if line.startswith("[")]) == 2


def test_search_reports_when_nothing_matches(
    examples_data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["search", "zebra", "--data", str(examples_data), "--retriever", "bm25"])

    assert code == 0
    assert "No matching chunks." in capsys.readouterr().out


def test_evaluate_prints_a_row_per_retriever(
    examples_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "evaluate",
            "--data",
            str(examples_dir / "data"),
            "--cases",
            str(examples_dir / "eval_cases.json"),
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    for name in ("dense", "tfidf", "bm25", "hybrid"):
        assert any(line.startswith(name) for line in output.splitlines())


def test_missing_data_directory_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["ask", "Anything?", "--data", str(tmp_path / "missing")])

    assert code == 1
    assert "raglab: error: Directory not found" in capsys.readouterr().err


def test_empty_data_directory_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["search", "Anything?", "--data", str(tmp_path)])

    assert code == 1
    assert "No .md or .txt files" in capsys.readouterr().err


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
