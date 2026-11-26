from pathlib import Path

import pytest

from raglab.chunking import chunk_documents
from raglab.models import Chunk, Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _document(source: str, text: str) -> Document:
    return Document(id=source, text=text, metadata={"source": source})


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        _document(
            "refunds.md",
            "You can request a full refund within 30 days of purchase. "
            "Refunds go back to the original payment method.",
        ),
        _document(
            "pricing.md",
            "The Plus plan costs 6 dollars per month. Students get 50 percent off the Plus plan.",
        ),
        _document(
            "security.md",
            "Two factor authentication is available on every plan. "
            "Notes are encrypted at rest with AES 256.",
        ),
        _document(
            "sync.md",
            "Sync conflicts happen when a note is edited on two devices while offline. "
            "Both versions are kept.",
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents: list[Document]) -> list[Chunk]:
    return chunk_documents(sample_documents)


@pytest.fixture
def examples_dir() -> Path:
    return PROJECT_ROOT / "examples"


@pytest.fixture
def examples_data(examples_dir: Path) -> Path:
    return examples_dir / "data"
