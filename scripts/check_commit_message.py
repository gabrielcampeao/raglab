from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)
SUBJECT_PATTERN = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([a-z0-9]+\))?: [a-z0-9](?:.*[^.\s])?$")
MAX_SUBJECT_LENGTH = 72
FORBIDDEN_CHARACTERS = ("-", "\u2013", "\u2014")
AUTOMATIC_PREFIXES = ("Merge ", "Revert ")


def validate(message: str) -> list[str]:
    lines = [line.rstrip() for line in message.splitlines() if not line.startswith("#")]
    subject = lines[0].strip() if lines else ""
    if not subject:
        return ["The commit subject is empty"]
    if subject.startswith(AUTOMATIC_PREFIXES):
        return []

    errors: list[str] = []
    if not SUBJECT_PATTERN.match(subject):
        errors.append(
            "Use 'type(scope): summary' with a known type, a lowercase scope, "
            "a lowercase summary and no trailing period"
        )
    if len(subject) > MAX_SUBJECT_LENGTH:
        errors.append(f"Keep the subject within {MAX_SUBJECT_LENGTH} characters")
    if any(character in subject for character in FORBIDDEN_CHARACTERS):
        errors.append("Write the subject without hyphens or dashes")
    if len(lines) > 1 and lines[1].strip():
        errors.append("Leave a blank line between the subject and the body")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: check_commit_message.py <message file>", file=sys.stderr)
        return 2
    errors = validate(Path(arguments[0]).read_text(encoding="utf-8"))
    for error in errors:
        print(f"commit message rejected: {error}", file=sys.stderr)
    if errors:
        print(f"allowed types: {', '.join(TYPES)}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
