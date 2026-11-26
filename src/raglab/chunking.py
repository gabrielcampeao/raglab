from __future__ import annotations

import re
from collections.abc import Iterable

from raglab.models import Chunk, Document

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_UNIT_SEPARATOR = "\n"


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


def chunk_text(text: str, max_chars: int = 500, overlap: int = 1) -> list[str]:
    _validate(max_chars, overlap)
    units: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > max_chars:
            units.extend(_split_long_sentence(sentence, max_chars))
        else:
            units.append(sentence)

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
