from collections.abc import Sequence
from pathlib import Path

import pytest

from raglab.evaluation import evaluate_retrieval, load_cases
from raglab.generation import Generator
from raglab.models import Document, SearchResult
from raglab.pipeline import RAGPipeline
from raglab.retrievers import create_retriever


class RecordingGenerator(Generator):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, question: str, prompt: str, results: Sequence[SearchResult]) -> str:
        self.prompts.append(prompt)
        return f"{len(results)} sources"


def test_pipeline_answers_from_ingested_documents(sample_documents: list[Document]) -> None:
    pipeline = RAGPipeline()

    assert pipeline.ingest(sample_documents) == 4

    answer = pipeline.ask("How many days do I have to request a refund?")

    assert "30 days" in answer.text
    assert answer.sources[0].chunk.source == "refunds.md"
    assert "Question: How many days do I have to request a refund?" in answer.prompt


def test_pipeline_passes_prompt_to_custom_generator(sample_documents: list[Document]) -> None:
    generator = RecordingGenerator()
    pipeline = RAGPipeline(generator=generator, top_k=2)
    pipeline.ingest(sample_documents)

    answer = pipeline.ask("Is two factor authentication available?")

    assert answer.text == "2 sources"
    assert generator.prompts == [answer.prompt]
    assert "security.md" in answer.prompt


def test_pipeline_accumulates_documents_across_ingest_calls(
    sample_documents: list[Document],
) -> None:
    pipeline = RAGPipeline(create_retriever("bm25"))
    pipeline.ingest(sample_documents[:2])
    pipeline.ingest(sample_documents[2:])

    assert len(pipeline.chunks) == 4
    assert pipeline.retrieve("sync conflicts offline")[0].chunk.source == "sync.md"
    assert pipeline.retrieve("refund")[0].chunk.source == "refunds.md"


def test_pipeline_requires_documents_before_answering() -> None:
    with pytest.raises(RuntimeError):
        RAGPipeline().ask("Anything?")


def test_pipeline_rejects_empty_questions(sample_documents: list[Document]) -> None:
    pipeline = RAGPipeline()
    pipeline.ingest(sample_documents)

    with pytest.raises(ValueError):
        pipeline.ask("   ")


@pytest.mark.parametrize("arguments", [{"top_k": 0}, {"max_chars": 0}, {"overlap": -1}])
def test_pipeline_rejects_invalid_settings(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        RAGPipeline(**arguments)


def test_pipeline_ingests_a_directory(examples_data: Path) -> None:
    pipeline = RAGPipeline()

    assert pipeline.ingest_directory(examples_data) > 0
    assert pipeline.ask("How do I force a full resync?").sources[0].chunk.source == "sync.md"


@pytest.mark.parametrize("name", ["bm25", "hybrid"])
def test_sample_knowledge_base_retrieval_quality(name: str, examples_dir: Path) -> None:
    pipeline = RAGPipeline(create_retriever(name))
    pipeline.ingest_directory(examples_dir / "data")
    cases = load_cases(examples_dir / "eval_cases.json")

    report = evaluate_retrieval(pipeline.retrieve, cases, k=3)

    assert report.hit_rate >= 0.9
    assert report.mrr >= 0.8
