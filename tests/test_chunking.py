import pytest

from raglab.chunking import chunk_documents, chunk_text, split_sentences
from raglab.models import Document


def test_split_sentences_breaks_on_punctuation_and_newlines() -> None:
    text = "# Title\nFirst one. Second one?\n\nThird!"
    assert split_sentences(text) == ["# Title", "First one.", "Second one?", "Third!"]


def test_chunk_text_respects_max_chars_and_keeps_every_sentence() -> None:
    sentences = [f"Sentence number {index} is here." for index in range(20)]
    chunks = chunk_text(" ".join(sentences), max_chars=80, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert [line for chunk in chunks for line in chunk.split("\n")] == sentences


def test_chunk_text_overlap_repeats_trailing_sentence() -> None:
    text = "Alpha one. Bravo two. Charlie three. Delta four."

    chunks = chunk_text(text, max_chars=25, overlap=1)

    assert chunks == [
        "Alpha one.\nBravo two.",
        "Bravo two.\nCharlie three.",
        "Delta four.",
    ]


def test_chunk_text_splits_sentences_longer_than_the_limit() -> None:
    text = " ".join(["word"] * 50)

    chunks = chunk_text(text, max_chars=40, overlap=0)

    assert all(len(chunk) <= 40 for chunk in chunks)
    assert sum(len(chunk.split()) for chunk in chunks) == 50


def test_chunk_text_of_blank_text_is_empty() -> None:
    assert chunk_text("   \n  ") == []


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (-5, 0), (100, -1)])
def test_chunk_text_rejects_invalid_arguments(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("Some text.", max_chars=max_chars, overlap=overlap)


def test_chunk_documents_assigns_ids_positions_and_metadata() -> None:
    documents = [Document(id="a.md", text="One. Two.", metadata={"source": "a.md"})]

    chunks = chunk_documents(documents, max_chars=5, overlap=0)

    assert [chunk.id for chunk in chunks] == ["a.md#0", "a.md#1"]
    assert [chunk.position for chunk in chunks] == [0, 1]
    assert [chunk.text for chunk in chunks] == ["One.", "Two."]
    assert all(chunk.document_id == "a.md" and chunk.source == "a.md" for chunk in chunks)


def test_chunk_source_falls_back_to_document_id() -> None:
    chunks = chunk_documents([Document(id="plain.txt", text="Hello there.")])

    assert chunks[0].source == "plain.txt"
