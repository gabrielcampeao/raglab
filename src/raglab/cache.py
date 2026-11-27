from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from raglab.embeddings import Embedder, HashingEmbedder
from raglab.retrievers import MetadataFilter
from raglab.vectormath import Vector, cosine_similarity, norm

if TYPE_CHECKING:
    from raglab.pipeline import Answer

ScopeKey = tuple[tuple[str, str], ...]


@dataclass
class _CacheEntry:
    vector: Vector
    created_at: float
    answer: Answer


def _canonical_scope(filters: MetadataFilter | None) -> ScopeKey:
    if not filters:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in filters.items()))


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        similarity_threshold: float = 0.92,
        max_entries: int = 256,
        ttl_seconds: float | None = None,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.embedder = embedder if embedder is not None else HashingEmbedder(dimensions=512)
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._buckets: dict[ScopeKey, list[_CacheEntry]] = {}

    def clear(self) -> None:
        self._buckets.clear()

    def _prune_expired(self, scope: ScopeKey, now: float) -> list[_CacheEntry]:
        entries = self._buckets.get(scope, [])
        if self.ttl_seconds is None or not entries:
            return entries
        cutoff = now - self.ttl_seconds
        fresh = [entry for entry in entries if entry.created_at >= cutoff]
        self._buckets[scope] = fresh
        return fresh

    def lookup(self, question: str, *, filters: MetadataFilter | None = None) -> Answer | None:
        start = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        if norm(query_vector) == 0.0:
            return None
        scope = _canonical_scope(filters)
        entries = self._prune_expired(scope, time.monotonic())
        best_entry: _CacheEntry | None = None
        best_score = -1.0
        for entry in entries:
            score = cosine_similarity(query_vector, entry.vector)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is None or best_score < self.similarity_threshold:
            return None
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        cached_trace = replace(
            best_entry.answer.trace,
            retrieval_ms=0.0,
            rerank_ms=0.0,
            generation_ms=0.0,
            total_ms=elapsed_ms,
            cache_hit=True,
        )
        return replace(best_entry.answer, question=question, trace=cached_trace)

    def store(
        self,
        question: str,
        answer: Answer,
        *,
        filters: MetadataFilter | None = None,
    ) -> None:
        query_vector = self.embedder.embed_query(question)
        if norm(query_vector) == 0.0:
            return
        scope = _canonical_scope(filters)
        bucket = self._buckets.setdefault(scope, [])
        bucket.append(_CacheEntry(vector=query_vector, created_at=time.monotonic(), answer=answer))
        if len(bucket) > self.max_entries:
            del bucket[0 : len(bucket) - self.max_entries]
