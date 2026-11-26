from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from raglab.models import Chunk, Document

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_UNIT_SEPARATOR = "\n"


@dataclass(frozen=True)
class MarkdownSection:
    heading_path: tuple[str, ...]
    body: str

    @property
    def label(self) -> str:
        return " > ".join(self.heading_path)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def _validate(max_chars: int, overlap: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _joined_length(window: list[str], unit: str) -> int:
    return len(_UNIT_SEPARATOR.join([*window, unit]))


def _pack_units(units: list[str], max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    window: list[str] = []
    for unit in units:
        if window and _joined_length(window, unit) > max_chars:
            chunks.append(_UNIT_SEPARATOR.join(window))
            window = window[-overlap:] if overlap else []
            while window and _joined_length(window, unit) > max_chars:
                window.pop(0)
        window.append(unit)
    if window:
        chunks.append(_UNIT_SEPARATOR.join(window))
    return chunks


def chunk_text(text: str, max_chars: int = 500, overlap: int = 1) -> list[str]:
    _validate(max_chars, overlap)
    units: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > max_chars:
            units.extend(_split_long_sentence(sentence, max_chars))
        else:
            units.append(sentence)
    return _pack_units(units, max_chars, overlap)


def split_markdown_sections(text: str) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            sections.append(MarkdownSection(tuple(title for _, title in heading_stack), body))

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if not in_fence:
            match = _HEADING_PATTERN.match(line)
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                continue
        buffer.append(line)

    flush()
    return sections


def _split_code_block(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    lines = block.splitlines()
    open_fence = lines[0].strip()
    has_closing_fence = len(lines) > 1 and lines[-1].lstrip().startswith("```")
    close_fence = "```"
    body_lines = lines[1:-1] if has_closing_fence else lines[1:]
    overhead = len(open_fence) + len(close_fence) + 2

    if max_chars > overhead + 10:
        inner_budget = max_chars - overhead
        inner_units: list[str] = []
        for raw_line in body_lines:
            preserved = raw_line.rstrip()
            if not preserved.strip():
                continue
            if len(preserved) > inner_budget:
                inner_units.extend(_split_long_sentence(preserved.strip(), inner_budget))
            else:
                inner_units.append(preserved)
        return [
            f"{open_fence}\n{piece}\n{close_fence}"
            for piece in _pack_units(inner_units, inner_budget, overlap=0)
        ]

    fallback_units: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if len(stripped) > max_chars:
            fallback_units.extend(_split_long_sentence(stripped, max_chars))
        else:
            fallback_units.append(stripped)
    return _pack_units(fallback_units, max_chars, overlap=0)


def _section_units(body: str, max_chars: int) -> list[str]:
    units: list[str] = []
    prose_lines: list[str] = []
    fence_lines: list[str] = []
    in_fence = False

    def flush_prose() -> None:
        prose = "\n".join(prose_lines).strip()
        prose_lines.clear()
        if not prose:
            return
        for sentence in split_sentences(prose):
            if len(sentence) > max_chars:
                units.extend(_split_long_sentence(sentence, max_chars))
            else:
                units.append(sentence)

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            if in_fence:
                fence_lines.append(line)
                units.extend(_split_code_block("\n".join(fence_lines), max_chars))
                fence_lines.clear()
                in_fence = False
            else:
                flush_prose()
                fence_lines.append(line)
                in_fence = True
        elif in_fence:
            fence_lines.append(line)
        else:
            prose_lines.append(line)

    if fence_lines:
        units.extend(_split_code_block("\n".join(fence_lines), max_chars))
    flush_prose()
    return units


def chunk_markdown(text: str, max_chars: int = 500, overlap: int = 1) -> list[tuple[str, str]]:
    _validate(max_chars, overlap)
    results: list[tuple[str, str]] = []
    for section in split_markdown_sections(text):
        units = _section_units(section.body, max_chars)
        for chunk in _pack_units(units, max_chars, overlap):
            results.append((section.label, chunk))
    return results


def chunk_documents(
    documents: Iterable[Document], max_chars: int = 500, overlap: int = 1
) -> list[Chunk]:
    _validate(max_chars, overlap)
    chunks: list[Chunk] = []
    for document in documents:
        for position, text in enumerate(chunk_text(document.text, max_chars, overlap)):
            chunks.append(
                Chunk(
                    id=f"{document.id}#{position}",
                    text=text,
                    document_id=document.id,
                    position=position,
                    metadata=dict(document.metadata),
                )
            )
    return chunks


def chunk_markdown_documents(
    documents: Iterable[Document], max_chars: int = 500, overlap: int = 1
) -> list[Chunk]:
    _validate(max_chars, overlap)
    chunks: list[Chunk] = []
    for document in documents:
        pairs = chunk_markdown(document.text, max_chars, overlap)
        for position, (section_label, text) in enumerate(pairs):
            metadata = dict(document.metadata)
            if section_label:
                metadata["section"] = section_label
            chunks.append(
                Chunk(
                    id=f"{document.id}#{position}",
                    text=text,
                    document_id=document.id,
                    position=position,
                    metadata=metadata,
                )
            )
    return chunks
