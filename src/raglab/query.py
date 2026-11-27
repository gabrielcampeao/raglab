from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping

from raglab.models import Chunk, SearchResult
from raglab.reranking import Reranker
from raglab.retrievers import MetadataFilter, Retriever

_WORD_PATTERN = re.compile(r"[^\W_]+")
_COMPOUND_SPLIT = re.compile(
    r"\?\s+(?=[A-Za-z])"
    r"|;\s+"
    r"|,?\s+(?:and|as well as|plus)\s+"
    r"(?=(?:how|what|why|when|where|which|who|can|should|do|does|is|are)\b)",
    re.IGNORECASE,
)

DEFAULT_ACRONYMS: dict[str, str] = {
    "oom": "out of memory exit code 137",
    "dlq": "dead letter queue",
    "mtls": "mutual tls certificate",
    "slo": "service level objective burn rate",
    "ttl": "time to live expiration",
    "wal": "write ahead log",
    "sso": "single sign on saml",
}


class QueryExpander(ABC):
    @abstractmethod
    def expand(self, query: str) -> list[str]:
        raise NotImplementedError


class AcronymQueryExpander(QueryExpander):
    def __init__(self, expansions: Mapping[str, str] | None = None) -> None:
        source = DEFAULT_ACRONYMS if expansions is None else expansions
        self.expansions = {key.lower(): value for key, value in source.items()}

    def expand(self, query: str) -> list[str]:
        clean = query.strip()
        if not clean:
            return []
        tokens = [tok.lower() for tok in _WORD_PATTERN.findall(clean)]
        matched = [self.expansions[tok] for tok in dict.fromkeys(tokens) if tok in self.expansions]
        if not matched:
            return [clean]
        return [clean, f"{clean} {' '.join(matched)}"]


def decompose_query(query: str) -> list[str]:
    clean = query.strip()
    if not clean:
        return []
    raw_parts = [
        part.strip().rstrip(",;:") for part in _COMPOUND_SPLIT.split(clean) if part.strip()
    ]
    if len(raw_parts) <= 1:
        return [clean]
    normalized: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        candidate = part if part.endswith("?") else f"{part}?"
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized if normalized else [clean]


class SubQuestionDecomposer(QueryExpander):
    def expand(self, query: str) -> list[str]:
        parts = decompose_query(query)
        if len(parts) <= 1:
            return [query.strip()] if query.strip() else []
        return parts


def _default_hypothesis(question: str) -> str:
    tokens = _WORD_PATTERN.findall(question)
    if not tokens:
        return ""
    return f"Operational procedure and root cause specification for {' '.join(tokens)}."


class HyDERetriever(Retriever):
    def __init__(
        self,
        retriever: Retriever,
        hypothesis_builder: Callable[[str], str] | None = None,
        *,
        rrf_k: int = 60,
        candidate_pool: int = 15,
    ) -> None:
        if rrf_k <= 0 or candidate_pool <= 0:
            raise ValueError("rrf_k and candidate_pool must be positive")
        self.retriever = retriever
        self.hypothesis_builder = hypothesis_builder or _default_hypothesis
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def index(self, chunks: Iterable[Chunk]) -> None:
        self.retriever.index(chunks)

    def retrieve(
        self, query: str, k: int = 3, *, filters: MetadataFilter | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        clean = query.strip()
        if not clean or not _WORD_PATTERN.findall(clean):
            return []
        draft = self.hypothesis_builder(clean).strip()
        probes = [clean, draft] if draft and draft != clean else [clean]
        pool = max(k, self.candidate_pool)
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for probe in probes:
            results = (
                self.retriever.retrieve(probe, pool, filters=filters)
                if filters is not None
                else self.retriever.retrieve(probe, pool)
            )
            for rank, item in enumerate(results, start=1):
                cid = item.chunk.id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                by_id[cid] = item.chunk
        merged = [SearchResult(chunk=by_id[cid], score=score) for cid, score in fused.items()]
        merged.sort(key=lambda item: (-item.score, item.chunk.id))
        return merged[:k]


def _interleave_variants(per_variant: list[list[SearchResult]], k: int) -> list[SearchResult]:
    interleaved: list[SearchResult] = []
    deferred: list[SearchResult] = []
    seen_sources: set[str] = set()
    seen_ids: set[str] = set()
    max_len = max((len(items) for items in per_variant), default=0)

    for depth in range(max_len):
        for items in per_variant:
            if depth >= len(items):
                continue
            candidate = items[depth]
            if candidate.chunk.id in seen_ids:
                continue
            if len(interleaved) < len(per_variant) and candidate.chunk.source in seen_sources:
                deferred.append(candidate)
                continue
            seen_ids.add(candidate.chunk.id)
            seen_sources.add(candidate.chunk.source)
            interleaved.append(candidate)
            if len(interleaved) >= k:
                break
        if len(interleaved) >= k:
            break

    if len(interleaved) < k:
        for candidate in deferred:
            if candidate.chunk.id in seen_ids:
                continue
            seen_ids.add(candidate.chunk.id)
            interleaved.append(candidate)
            if len(interleaved) >= k:
                break

    return [
        SearchResult(chunk=item.chunk, score=1.0 / (1.0 + index))
        for index, item in enumerate(interleaved[:k])
    ]


class MultiQueryRetriever(Retriever):
    def __init__(
        self,
        retriever: Retriever,
        expander: QueryExpander | None = None,
        *,
        fusion: str | None = None,
        rrf_k: int = 60,
        candidate_pool: int = 15,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        resolved_expander = expander if expander is not None else AcronymQueryExpander()
        resolved_fusion = (
            fusion
            if fusion is not None
            else ("interleave" if isinstance(resolved_expander, SubQuestionDecomposer) else "rrf")
        )
        if resolved_fusion not in ("rrf", "interleave"):
            raise ValueError("fusion must be 'rrf' or 'interleave'")
        self.retriever = retriever
        self.expander = resolved_expander
        self.fusion = resolved_fusion
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def index(self, chunks: Iterable[Chunk]) -> None:
        self.retriever.index(chunks)

    def retrieve_decomposed(
        self,
        query: str,
        k: int = 3,
        *,
        filters: MetadataFilter | None = None,
        reranker: Reranker | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        variants = self.expander.expand(query)
        if not variants:
            return []
        pool = max(k, self.candidate_pool)
        per_variant: list[list[SearchResult]] = []
        for variant in variants:
            raw = (
                self.retriever.retrieve(variant, pool, filters=filters)
                if filters is not None
                else self.retriever.retrieve(variant, pool)
            )
            ranked = reranker.rerank(variant, raw) if reranker is not None and raw else raw
            per_variant.append(ranked)

        if self.fusion == "interleave" and len(per_variant) > 1:
            return _interleave_variants(per_variant, k)

        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for results in per_variant:
            for rank, item in enumerate(results, start=1):
                cid = item.chunk.id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                by_id[cid] = item.chunk
        merged = [SearchResult(chunk=by_id[cid], score=score) for cid, score in fused.items()]
        merged.sort(key=lambda item: (-item.score, item.chunk.id))
        return merged[:k]

    def retrieve(
        self, query: str, k: int = 3, *, filters: MetadataFilter | None = None
    ) -> list[SearchResult]:
        return self.retrieve_decomposed(query, k, filters=filters)
