# Contributing

Contributions that sharpen retrieval accuracy, expand benchmark coverage, or clarify mathematical explanations are welcome.

## Environment setup

```bash
python -m venv .venv
pip install -e ".[dev]"
git config core.hooksPath .githooks
git config commit.template .gitmessage
```

Activate the virtual environment before running `pip`. The two `git config` commands enable the local commit message validator and commit template.

## Verification gate before opening a pull request

```bash
ruff format --check .
ruff check .
pytest --cov
python scripts/generate_charts.py
```

Continuous integration runs `ruff` and the full `pytest` suite across Python 3.10, 3.11, 3.12, and 3.13 on every push and pull request.

## Engineering conventions

* **Zero comments or docstrings in Python files:** Express intent through module boundaries, small functions, and explicit type annotations.
* **Zero runtime dependencies:** Keep `dependencies = []` in `pyproject.toml`. External model integrations belong behind optional adapters using the standard library or user supplied callables.
* **Test driven changes:** Every new feature or bug fix starts with a failing test in `tests/`. If benchmark numbers change, rerun `python scripts/generate_charts.py` so `benchmark_results.json` and `docs/assets/*.svg` stay synchronized.

## Commit message standard

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
type(scope): concise lowercase summary
```

Allowed types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, `test`.

`.githooks/commit-msg` enforces four rules on every commit:

* Subject length of at most 72 characters with no trailing period.
* Lowercase type, optional lowercase alphanumeric scope, and summary starting with a lowercase letter or digit.
* No hyphens, en dashes, or em dashes in the subject line.
* A blank line separating the subject from an optional body explaining why the change was made.
