from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from raglab.models import Chunk, SearchResult
from raglab.retrievers import MetadataFilter, Retriever

_WORD_PATTERN = re.compile(r"[^\W_]+")

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


class MultiQueryRetriever(Retriever):
    def __init__(
        self,
        retriever: Retriever,
        expander: QueryExpander | None = None,
        *,
        rrf_k: int = 60,
        candidate_pool: int = 15,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        self.retriever = retriever
        self.expander = expander if expander is not None else AcronymQueryExpander()
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def index(self, chunks: Iterable[Chunk]) -> None:
        self.retriever.index(chunks)

    def retrieve(
        self, query: str, k: int = 3, *, filters: MetadataFilter | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        variants = self.expander.expand(query)
        if not variants:
            return []
        pool = max(k, self.candidate_pool)
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for variant in variants:
            results = (
                self.retriever.retrieve(variant, pool, filters=filters)
                if filters is not None
                else self.retriever.retrieve(variant, pool)
            )
            for rank, item in enumerate(results, start=1):
                cid = item.chunk.id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                by_id[cid] = item.chunk
        merged = [SearchResult(chunk=by_id[cid], score=score) for cid, score in fused.items()]
        merged.sort(key=lambda item: (-item.score, item.chunk.id))
        return merged[:k]
