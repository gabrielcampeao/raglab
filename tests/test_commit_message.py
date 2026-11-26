import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_commit_message.py"


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_commit_message", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "message",
    [
        "feat(retrieval): add hybrid retriever",
        "fix: handle empty documents",
        "docs: explain how rag works\n\nThe guide walks through every stage.",
        "test(cli): cover the evaluate command\n",
        "Merge branch 'main' into feature",
    ],
)
def test_valid_messages_pass(checker: ModuleType, message: str) -> None:
    assert checker.validate(message) == []


@pytest.mark.parametrize(
    "message",
    [
        "",
        "added some stuff",
        "Feat: add thing",
        "feat: Add thing",
        "feat: add thing.",
        "feat(Retrieval): add thing",
        "feature: add thing",
        "feat: add an end-to-end test",
        "feat: add thing \u2014 now",
        "feat: add thing\nbody without a blank line",
        "feat: " + "x" * 80,
    ],
)
def test_invalid_messages_fail(checker: ModuleType, message: str) -> None:
    assert checker.validate(message)


def test_comment_lines_are_ignored(checker: ModuleType) -> None:
    assert checker.validate("# Please enter the commit message\nchore: tidy up") == []


def test_main_reads_message_file(checker: ModuleType, tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.txt"
    good.write_text("ci: run tests on pull requests\n", encoding="utf-8")
    bad.write_text("Updated files\n", encoding="utf-8")

    assert checker.main([str(good)]) == 0
    assert checker.main([str(bad)]) == 1
    assert checker.main([]) == 2
