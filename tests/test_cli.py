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


def test_ask_supports_markdown_rerank_mmr_budget_and_trace(
    examples_data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "ask",
            "How long do I have to request a refund?",
            "--data",
            str(examples_data),
            "--markdown",
            "--rerank",
            "--mmr",
            "0.75",
            "--max-tokens",
            "180",
            "--show-trace",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "30 days" in output
    assert "Trace:" in output
    assert "Grounding:" in output


def test_cli_rejects_malformed_filter_key_or_value(
    examples_data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["search", "refund", "--data", str(examples_data), "--filter", "kind="]) == 1
    assert "Invalid filter" in capsys.readouterr().err


def test_cli_evaluate_on_incident_copilot_reproduces_scoped_triage_hits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[1] / "projects" / "incident_copilot"
    code = main(
        [
            "evaluate",
            "--data",
            str(root / "corpus"),
            "--cases",
            str(root / "benchmark.json"),
            "--markdown",
            "--rerank",
            "--mmr",
            "0.65",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    hybrid_line = next(line for line in output.splitlines() if line.startswith("hybrid"))
    assert "1.00" in hybrid_line


def test_cli_ask_runs_against_qdrant_backend(capsys: pytest.CaptureFixture[str]) -> None:
    corpus = Path(__file__).resolve().parents[1] / "projects" / "incident_copilot" / "corpus"
    question = (
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?"
    )
    code = main(
        [
            "ask",
            question,
            "--data",
            str(corpus),
            "--backend",
            "qdrant",
            "--filter",
            "kind=runbook",
            "--markdown",
            "--rerank",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "transaction" in output
    assert "postgres_pool_saturation.md" in output


def test_cli_ask_runs_against_hnsw_backend(capsys: pytest.CaptureFixture[str]) -> None:
    corpus = Path(__file__).resolve().parents[1] / "projects" / "incident_copilot" / "corpus"
    question = (
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?"
    )
    code = main(
        [
            "ask",
            question,
            "--data",
            str(corpus),
            "--backend",
            "hnsw",
            "--filter",
            "kind=runbook",
            "--markdown",
            "--rerank",
            "--show-trace",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "transaction" in output
    assert "postgres_pool_saturation.md" in output
    assert "Trace:" in output

    search_code = main(
        [
            "search",
            "SQLSTATE 53300 PgBouncer",
            "--data",
            str(corpus),
            "--retriever",
            "dense",
            "--backend",
            "hnsw",
            "--k",
            "2",
        ]
    )
    search_output = capsys.readouterr().out
    assert search_code == 0
    assert "postgres_pool_saturation.md" in search_output
