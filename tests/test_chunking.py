import pytest

from raglab.chunking import (
    MarkdownSection,
    chunk_documents,
    chunk_markdown,
    chunk_markdown_documents,
    chunk_text,
    split_markdown_sections,
    split_sentences,
)
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


def test_split_markdown_sections_tracks_heading_hierarchy() -> None:
    markdown = (
        "# Postgres saturation\n"
        "Overview paragraph.\n\n"
        "## Symptoms\n"
        "Clients see SQLSTATE 53300.\n\n"
        "### Metrics\n"
        "Active connections reach max_connections.\n\n"
        "## Mitigation\n"
        "Enable PgBouncer transaction pooling.\n"
    )

    sections = split_markdown_sections(markdown)

    assert sections == [
        MarkdownSection(("Postgres saturation",), "Overview paragraph."),
        MarkdownSection(("Postgres saturation", "Symptoms"), "Clients see SQLSTATE 53300."),
        MarkdownSection(
            ("Postgres saturation", "Symptoms", "Metrics"),
            "Active connections reach max_connections.",
        ),
        MarkdownSection(
            ("Postgres saturation", "Mitigation"),
            "Enable PgBouncer transaction pooling.",
        ),
    ]


def test_chunk_markdown_keeps_fenced_code_blocks_intact_when_within_limit() -> None:
    markdown = (
        "# Runbook\n"
        "## Diagnosis\n"
        "Run this query first.\n\n"
        "```sql\n"
        "SELECT state, count(*)\n"
        "FROM pg_stat_activity\n"
        "GROUP BY state;\n"
        "```\n\n"
        "Then inspect idle transactions.\n"
    )

    pairs = chunk_markdown(markdown, max_chars=140, overlap=0)

    code_chunks = [text for _, text in pairs if "```sql" in text]
    assert len(code_chunks) == 1
    assert "SELECT state, count(*)" in code_chunks[0]
    assert "GROUP BY state;\n```" in code_chunks[0]


def test_chunk_markdown_splits_oversized_fenced_code_blocks_safely() -> None:
    lines = "\n".join(f"SELECT column_{index} FROM ledger;" for index in range(12))
    markdown = f"# Runbook\n```sql\n{lines}\n```"

    pairs = chunk_markdown(markdown, max_chars=90, overlap=0)

    assert len(pairs) > 1
    assert all(len(text) <= 90 for _, text in pairs)


def test_chunk_markdown_documents_populates_section_metadata() -> None:
    doc = Document(
        id="runbooks/pg.md",
        text="# Pool saturation\n## Mitigation\nSwitch PgBouncer to transaction mode.",
        metadata={"source": "runbooks/pg.md", "kind": "runbook"},
    )

    chunks = chunk_markdown_documents([doc], max_chars=200, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].metadata == {
        "source": "runbooks/pg.md",
        "kind": "runbook",
        "section": "Pool saturation > Mitigation",
    }
    assert "Switch PgBouncer to transaction mode." in chunks[0].text


def test_chunk_markdown_rewraps_fences_when_splitting_oversized_code_blocks() -> None:
    lines = "\n".join(f"  SELECT col_{index} FROM ledger;" for index in range(10))
    markdown = f"# Runbook\n```sql\n{lines}\n```"

    pairs = chunk_markdown(markdown, max_chars=110, overlap=0)

    assert len(pairs) > 1
    for _, text in pairs:
        assert text.startswith("```sql\n")
        assert text.endswith("\n```")
        assert "  SELECT " in text
        assert len(text) <= 110
