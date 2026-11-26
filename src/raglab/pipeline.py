from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from raglab.chunking import chunk_documents
from raglab.documents import load_directory
from raglab.generation import ExtractiveGenerator, Generator
from raglab.models import Chunk, Document, SearchResult
from raglab.prompting import DEFAULT_TEMPLATE, build_prompt
from raglab.retrievers import HybridRetriever, Retriever


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: tuple[SearchResult, ...]
    prompt: str


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        *,
        max_chars: int = 500,
        overlap: int = 1,
        top_k: int = 3,
        template: str = DEFAULT_TEMPLATE,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.generator = generator if generator is not None else ExtractiveGenerator()
        self.max_chars = max_chars
        self.overlap = overlap
        self.top_k = top_k
        self.template = template
        self._chunks: list[Chunk] = []

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def ingest(self, documents: Iterable[Document]) -> int:
        new_chunks = chunk_documents(documents, self.max_chars, self.overlap)
        self._chunks.extend(new_chunks)
        self.retriever.index(self._chunks)
        return len(new_chunks)

    def ingest_directory(self, path: str | Path) -> int:
        return self.ingest(load_directory(path))

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("The question must not be empty")
        if not self._chunks:
            raise RuntimeError("Ingest documents before asking questions")
        return self.retriever.retrieve(question, k or self.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        results = self.retrieve(question, k)
        prompt = build_prompt(question, results, self.template)
        text = self.generator.generate(question, prompt, results)
        return Answer(question=question, text=text, sources=tuple(results), prompt=prompt)
