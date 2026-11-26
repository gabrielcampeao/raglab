import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(path.name for path in EXAMPLES_DIR.glob("*.py"))


def test_examples_are_present() -> None:
    assert len(SCRIPTS) >= 5


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs_successfully(script: str) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
    }

    completed = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
