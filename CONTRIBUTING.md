# Contributing

raglab is a study project, so an issue that points out a confusing explanation is as welcome as a code change.

## Setting up

```bash
python -m venv .venv
pip install -e ".[dev]"
git config core.hooksPath .githooks
git config commit.template .gitmessage
```

Activate the virtual environment before running `pip`. The two `git config` lines turn on the commit message check and the message template for your clone.

## Before opening a pull request

```bash
ruff format .
ruff check .
pytest
```

CI runs the same checks on Python 3.10, 3.11, 3.12 and 3.13.

## Code style

* The code explains itself through names and type hints, so the project uses no comments or docstrings. When a block needs explaining, extract it into a well named function.
* The package has no runtime dependencies. Every algorithm should stay visible in this repository.
* Every change in behavior comes with a test.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
type(scope): short summary in lowercase
```

| Type | Use it for |
| --- | --- |
| `feat` | new behavior |
| `fix` | a bug fix |
| `docs` | documentation only |
| `test` | adding or correcting tests |
| `refactor` | code changes that keep behavior the same |
| `perf` | speed or memory improvements |
| `build` | packaging and dependencies |
| `ci` | GitHub Actions |
| `style` | formatting only |
| `chore` | maintenance that fits nowhere else |
| `revert` | undoing an earlier commit |

The scope is optional and names the area touched, such as `chunking`, `retrieval` or `cli`. The hook in `.githooks/commit-msg` rejects a message unless:

* the subject has at most 72 characters and does not end with a period
* the type and scope are lowercase and the summary starts with a lowercase letter or digit
* the subject contains no hyphens or dashes
* a blank line separates the subject from the body

Use the body to explain why the change was made. Some examples:

```text
feat(retrieval): add reciprocal rank fusion
fix(chunking): keep headings out of extracted answers
docs: explain how bm25 scores documents
```
