from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

from raglab.embeddings import Embedder, HashingEmbedder, TfidfEmbedder
from raglab.models import Chunk, SearchResult
from raglab.text import tokenize
from raglab.vectorstore import InMemoryVectorStore

RETRIEVER_NAMES = ("dense", "tfidf", "bm25", "hybrid")


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def _rank(results: list[SearchResult], k: int) -> list[SearchResult]:
    results.sort(key=lambda result: (-result.score, result.chunk.id))
    return results[:k]


class Retriever(ABC):
    @abstractmethod
    def index(self, chunks: Iterable[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> list[SearchResult]:
        raise NotImplementedError


class DenseRetriever(Retriever):
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: InMemoryVectorStore | None = None,
        *,
        min_score: float = 0.0,
    ) -> None:
        self.embedder = embedder if embedder is not None else HashingEmbedder()
        self.store = store if store is not None else InMemoryVectorStore()
        self.min_score = min_score

    def index(self, chunks: Iterable[Chunk]) -> None:
        chunk_list = list(chunks)
        texts = [chunk.text for chunk in chunk_list]
        self.embedder.fit(texts)
        self.store.clear()
        if chunk_list:
            self.store.add(chunk_list, self.embedder.embed(texts))

    def retrieve(self, query: str, k: int = 3) -> list[SearchResult]:
        _check_k(k)
        if len(self.store) == 0:
            return []
        results = self.store.search(self.embedder.embed_query(query), k)
        return [result for result in results if result.score > self.min_score]


class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 cannot be negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._average_length = 0.0
        self._idf: dict[str, float] = {}

    def index(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        self._term_frequencies = [Counter(tokenize(chunk.text)) for chunk in self._chunks]
        self._lengths = [sum(frequencies.values()) for frequencies in self._term_frequencies]
        total = len(self._chunks)
        self._average_length = sum(self._lengths) / total if total else 0.0
        document_frequency = Counter(
            term for frequencies in self._term_frequencies for term in frequencies
        )
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def score(self, query_terms: Sequence[str], position: int) -> float:
        frequencies = self._term_frequencies[position]
        if self._average_length == 0:
            return 0.0
        length_ratio = self._lengths[position] / self._average_length
        total = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            saturation = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
            total += self._idf[term] * frequency * (self.k1 + 1) / saturation
        return total

    def retrieve(self, query: str, k: int = 3) -> list[SearchResult]:
        _check_k(k)
        terms = tokenize(query)
        results: list[SearchResult] = []
        for position, chunk in enumerate(self._chunks):
            value = self.score(terms, position)
            if value > 0:
                results.append(SearchResult(chunk=chunk, score=value))
        return _rank(results, k)


class HybridRetriever(Retriever):
    def __init__(
        self,
        retrievers: Sequence[Retriever] | None = None,
        *,
        rrf_k: int = 60,
        candidate_pool: int = 20,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        self.retrievers = (
            list(retrievers) if retrievers else [DenseRetriever(HashingEmbedder()), BM25Retriever()]
        )
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def index(self, chunks: Iterable[Chunk]) -> None:
        chunk_list = list(chunks)
        for retriever in self.retrievers:
            retriever.index(chunk_list)

    def retrieve(self, query: str, k: int = 3) -> list[SearchResult]:
        _check_k(k)
        pool = max(k, self.candidate_pool)
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, result in enumerate(retriever.retrieve(query, pool), start=1):
                chunk_id = result.chunk.id
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
                chunks[chunk_id] = result.chunk
        results = [
            SearchResult(chunk=chunks[chunk_id], score=score) for chunk_id, score in fused.items()
        ]
        return _rank(results, k)


_FACTORIES: dict[str, Callable[[], Retriever]] = {
    "dense": lambda: DenseRetriever(HashingEmbedder()),
    "tfidf": lambda: DenseRetriever(TfidfEmbedder()),
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
}


def create_retriever(name: str) -> Retriever:
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown retriever '{name}'. Choose one of: {', '.join(RETRIEVER_NAMES)}")
    return factory()
