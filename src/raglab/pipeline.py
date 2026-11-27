from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from raglab.cache import SemanticCache
from raglab.chunking import chunk_documents, chunk_markdown_documents
from raglab.documents import load_directory
from raglab.generation import ExtractiveGenerator, Generator
from raglab.models import Chunk, Document, SearchResult
from raglab.parent_child import ParentChildRetriever
from raglab.prompting import (
    DEFAULT_TEMPLATE,
    build_prompt,
    estimate_tokens,
    format_context,
    pack_context,
    reorder_lost_in_middle,
)
from raglab.query import MultiQueryRetriever
from raglab.reranking import Reranker, diversify_mmr
from raglab.retrievers import HybridRetriever, MetadataFilter, Retriever
from raglab.security import ContextSanitizer
from raglab.telemetry import TraceSink, build_trace_event
from raglab.verification import GroundingReport, verify_answer


@dataclass(frozen=True)
class PipelineTrace:
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float
    candidates_retrieved: int
    candidates_selected: int
    context_tokens: int
    cache_hit: bool = False
    ranked_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: tuple[SearchResult, ...]
    prompt: str
    grounding: GroundingReport
    trace: PipelineTrace


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        *,
        reranker: Reranker | None = None,
        sanitizer: ContextSanitizer | None = None,
        cache: SemanticCache | None = None,
        trace_sink: TraceSink | None = None,
        candidate_pool: int = 15,
        mmr_lambda: float | None = None,
        max_context_tokens: int | None = None,
        reorder_context: bool = False,
        markdown_aware: bool = False,
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
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.generator = generator if generator is not None else ExtractiveGenerator()
        self.reranker = reranker
        self.sanitizer = sanitizer
        self.cache = cache
        self.trace_sink = trace_sink
        self.candidate_pool = candidate_pool
        self.mmr_lambda = mmr_lambda
        self.max_context_tokens = max_context_tokens
        self.reorder_context = reorder_context
        self.markdown_aware = markdown_aware
        self.max_chars = max_chars
        self.overlap = overlap
        self.top_k = top_k
        self.template = template
        self._chunks: list[Chunk] = []

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def ingest(self, documents: Iterable[Document]) -> int:
        cleaned_docs = (
            [self.sanitizer.clean_document(doc) for doc in documents]
            if self.sanitizer is not None
            else list(documents)
        )
        if isinstance(self.retriever, ParentChildRetriever):
            new_chunks = self.retriever.chunk_parents(cleaned_docs)
        else:
            chunker = chunk_markdown_documents if self.markdown_aware else chunk_documents
            new_chunks = chunker(cleaned_docs, self.max_chars, self.overlap)
        self._chunks.extend(new_chunks)
        self.retriever.index(self._chunks)
        if self.cache is not None:
            self.cache.clear()
        return len(new_chunks)

    def ingest_directory(self, path: str | Path) -> int:
        return self.ingest(load_directory(path))

    def _select_candidates(
        self,
        question: str,
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
    ) -> tuple[list[SearchResult], float, float, int]:
        if not question.strip():
            raise ValueError("The question must not be empty")
        if not self._chunks:
            raise RuntimeError("Ingest documents before asking questions")
        limit = self.top_k if k is None else k
        if limit <= 0:
            raise ValueError("k must be positive")

        delegated_interleave = (
            isinstance(self.retriever, MultiQueryRetriever)
            and self.retriever.fusion == "interleave"
        )
        uses_post_stage = (
            (self.reranker is not None and not delegated_interleave)
            or self.mmr_lambda is not None
            or self.sanitizer is not None
        )
        pool = max(limit, self.candidate_pool) if uses_post_stage else limit

        start_retrieval = time.perf_counter()
        if isinstance(self.retriever, MultiQueryRetriever) and delegated_interleave:
            raw_results = self.retriever.retrieve_decomposed(
                question,
                pool,
                filters=filters,
                reranker=self.reranker,
            )
        else:
            raw_results = (
                self.retriever.retrieve(question, pool, filters=filters)
                if filters is not None
                else self.retriever.retrieve(question, pool)
            )
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0

        start_rerank = time.perf_counter()
        selected = (
            self.sanitizer.sanitize_results(raw_results)
            if self.sanitizer is not None
            else list(raw_results)
        )
        if self.reranker is not None and not delegated_interleave and selected:
            rerank_limit = None if self.mmr_lambda is not None else limit
            selected = self.reranker.rerank(question, selected, k=rerank_limit)
        if self.mmr_lambda is not None and selected:
            selected = diversify_mmr(selected, k=limit, lambda_weight=self.mmr_lambda)
        else:
            selected = selected[:limit]
        rerank_ms = (time.perf_counter() - start_rerank) * 1000.0

        return selected, retrieval_ms, rerank_ms, len(raw_results)

    def retrieve(
        self,
        question: str,
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        selected, _, _, _ = self._select_candidates(question, k, filters=filters)
        return selected

    def ask(
        self,
        question: str,
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
        retrieval_query: str | None = None,
    ) -> Answer:
        if not question.strip():
            raise ValueError("The question must not be empty")
        if not self._chunks:
            raise RuntimeError("Ingest documents before asking questions")
        limit = self.top_k if k is None else k
        if limit <= 0:
            raise ValueError("k must be positive")

        probe = retrieval_query if retrieval_query is not None else question
        if self.cache is not None and k is None:
            cached = self.cache.lookup(probe, filters=filters)
            if cached is not None:
                if self.trace_sink is not None:
                    self.trace_sink.record(build_trace_event(cached, filters))
                return cached

        ask_start = time.perf_counter()
        selected, retrieval_ms, rerank_ms, raw_count = self._select_candidates(
            probe, k, filters=filters
        )
        if self.max_context_tokens is not None and selected:
            selected = pack_context(selected, self.max_context_tokens)
        ranked_sources = tuple(item.chunk.source for item in selected)
        if self.reorder_context and len(selected) > 2:
            selected = reorder_lost_in_middle(selected)
        formatted_context = format_context(selected)
        prompt = build_prompt(question, selected, self.template)
        context_tokens = estimate_tokens(formatted_context) if selected else 0

        start_generation = time.perf_counter()
        text = self.generator.generate(question, prompt, selected)
        generation_ms = (time.perf_counter() - start_generation) * 1000.0

        grounding = verify_answer(text, selected)
        total_ms = (time.perf_counter() - ask_start) * 1000.0
        trace = PipelineTrace(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            candidates_retrieved=raw_count,
            candidates_selected=len(selected),
            context_tokens=context_tokens,
            cache_hit=False,
            ranked_sources=ranked_sources,
        )
        answer = Answer(
            question=question,
            text=text,
            sources=tuple(selected),
            prompt=prompt,
            grounding=grounding,
            trace=trace,
        )
        if self.cache is not None and k is None:
            self.cache.store(probe, answer, filters=filters)
        if self.trace_sink is not None:
            self.trace_sink.record(build_trace_event(answer, filters))
        return answer
