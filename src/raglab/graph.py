from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from raglab.models import Chunk, SearchResult
from raglab.retrievers import MetadataFilter, Retriever, matches_filters

_BACKTICK_PATTERN = re.compile(r"`([^`\n]{2,48})`")
_CODE_PAIR_PATTERN = re.compile(
    r"\b(?:SQLSTATE|HTTP|TLS|AES|CVE|SEV|EXIT CODE)\s+[A-Z0-9._-]+\b",
    re.IGNORECASE,
)
_IDENTIFIER_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:[._][a-z0-9]+)+\b")
_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,8}\b")
_PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}\b")

_STOP_ENTITIES = frozenset(
    {
        "the",
        "when",
        "what",
        "how",
        "why",
        "where",
        "which",
        "who",
        "switch",
        "enable",
        "check",
        "page",
        "run",
        "all",
        "this",
        "that",
        "these",
        "those",
        "every",
        "before",
        "after",
        "both",
        "one",
        "two",
        "three",
        "any",
        "most",
        "some",
        "more",
        "only",
        "you",
        "your",
        "our",
        "workers",
        "clients",
        "pods",
        "notes",
        "data",
        "read",
        "write",
        "set",
        "add",
        "use",
        "from",
        "select",
        "group",
        "order",
        "not",
        "get",
        "by",
        "and",
        "or",
        "as",
        "in",
        "is",
        "on",
        "to",
        "of",
        "if",
        "do",
        "no",
        "for",
        "put",
        "post",
        "delete",
        "head",
        "null",
        "true",
        "false",
        "into",
        "join",
        "left",
        "right",
        "inner",
        "outer",
        "like",
        "desc",
        "asc",
        "limit",
        "with",
        "case",
        "then",
        "else",
        "end",
        "create",
        "drop",
        "alter",
        "table",
        "index",
        "values",
        "update",
        "insert",
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "now",
        "utc",
        "may",
        "can",
        "are",
        "was",
        "has",
        "have",
        "had",
        "did",
        "does",
        "will",
        "would",
        "should",
        "could",
        "symptoms",
        "diagnosis",
        "mitigation",
        "immediate",
        "escalation",
        "recovery",
        "corrective",
        "actions",
        "incident",
        "summary",
        "root",
        "cause",
        "context",
        "architecture",
        "record",
        "postmortem",
        "runbook",
    }
)


def _normalize_entity(raw: str) -> str:
    return " ".join(raw.strip().strip(".,;:!?()").lower().split())


def extract_entities(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def register(raw: str) -> None:
        cleaned = _normalize_entity(raw)
        if len(cleaned) < 2 or cleaned in _STOP_ENTITIES or cleaned in seen:
            return
        seen.add(cleaned)
        found.append(cleaned)

    for match in _BACKTICK_PATTERN.finditer(text):
        register(match.group(1))
    for match in _CODE_PAIR_PATTERN.finditer(text):
        register(match.group(0))
    for match in _IDENTIFIER_PATTERN.finditer(text):
        register(match.group(0))
    for match in _ACRONYM_PATTERN.finditer(text):
        register(match.group(0))
    for match in _PROPER_NOUN_PATTERN.finditer(text):
        register(match.group(0))
    return found


class KnowledgeGraph:
    def __init__(self) -> None:
        self._entity_chunks: dict[str, set[str]] = {}
        self._chunk_entities: dict[str, tuple[str, ...]] = {}
        self._adjacency: dict[str, dict[str, float]] = {}

    @property
    def entities(self) -> tuple[str, ...]:
        return tuple(sorted(self._entity_chunks))

    def _remove_chunk(self, chunk_id: str) -> None:
        previous = self._chunk_entities.pop(chunk_id, ())
        if not previous:
            return
        for idx, left in enumerate(previous):
            for right in previous[idx + 1 :]:
                left_edges = self._adjacency.get(left)
                if left_edges and right in left_edges:
                    if left_edges[right] <= 1.0:
                        del left_edges[right]
                    else:
                        left_edges[right] -= 1.0
                right_edges = self._adjacency.get(right)
                if right_edges and left in right_edges:
                    if right_edges[left] <= 1.0:
                        del right_edges[left]
                    else:
                        right_edges[left] -= 1.0
        for entity in previous:
            members = self._entity_chunks.get(entity)
            if members is not None:
                members.discard(chunk_id)
                if not members:
                    del self._entity_chunks[entity]
                    self._adjacency.pop(entity, None)

    def add_chunk(self, chunk_id: str, entities: Sequence[str]) -> None:
        if chunk_id in self._chunk_entities:
            self._remove_chunk(chunk_id)
        unique = tuple(
            dict.fromkeys(cleaned for entity in entities if (cleaned := _normalize_entity(entity)))
        )
        self._chunk_entities[chunk_id] = unique
        for entity in unique:
            self._entity_chunks.setdefault(entity, set()).add(chunk_id)
            self._adjacency.setdefault(entity, {})
        for idx, left in enumerate(unique):
            for right in unique[idx + 1 :]:
                self._adjacency[left][right] = self._adjacency[left].get(right, 0.0) + 1.0
                self._adjacency[right][left] = self._adjacency[right].get(left, 0.0) + 1.0

    def chunk_entities(self, chunk_id: str) -> tuple[str, ...]:
        return self._chunk_entities.get(chunk_id, ())

    def neighbors(self, entity: str) -> tuple[str, ...]:
        edges = self._adjacency.get(_normalize_entity(entity), {})
        ordered = sorted(edges.items(), key=lambda pair: (-pair[1], pair[0]))
        return tuple(neighbor for neighbor, _ in ordered)

    def multi_hop_scores(
        self,
        seeds: Iterable[str],
        *,
        hops: int = 2,
        decay: float = 0.45,
    ) -> dict[str, float]:
        if hops < 0:
            raise ValueError("hops cannot be negative")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        active: dict[str, float] = {}
        for seed in seeds:
            key = _normalize_entity(seed)
            if key in self._entity_chunks:
                active[key] = 1.0
        if not active or hops <= 0:
            return active

        frontier = dict(active)
        for _ in range(hops):
            next_frontier: dict[str, float] = {}
            for node, energy in frontier.items():
                edges = self._adjacency.get(node, {})
                if not edges:
                    continue
                max_weight = max(edges.values())
                for neighbor, weight in edges.items():
                    propagated = energy * decay * (weight / max_weight)
                    if propagated > active.get(neighbor, 0.0):
                        active[neighbor] = propagated
                        if propagated > next_frontier.get(neighbor, 0.0):
                            next_frontier[neighbor] = propagated
            frontier = next_frontier
            if not frontier:
                break
        return active


class EntityGraphRetriever(Retriever):
    def __init__(self, *, hops: int = 2, decay: float = 0.45) -> None:
        if hops < 0:
            raise ValueError("hops cannot be negative")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.hops = hops
        self.decay = decay
        self.graph = KnowledgeGraph()
        self._chunks: list[Chunk] = []
        self._idf: dict[str, float] = {}

    def index(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        self.graph = KnowledgeGraph()
        doc_freq: Counter[str] = Counter()
        for chunk in self._chunks:
            section = chunk.metadata.get("section", "")
            text = f"{section}\n{chunk.text}" if section else chunk.text
            entities = extract_entities(text)
            self.graph.add_chunk(chunk.id, entities)
            for entity in set(entities):
                doc_freq[entity] += 1
        total = len(self._chunks)
        self._idf = {
            entity: math.log((1.0 + total) / (1.0 + count)) + 1.0
            for entity, count in doc_freq.items()
        }

    def _query_seeds(self, query: str) -> list[str]:
        seeds = extract_entities(query)
        lowered = query.lower()
        for entity in self.graph.entities:
            if entity in seeds or len(entity) < 3:
                continue
            if re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", lowered):
                seeds.append(entity)
        return seeds

    def retrieve(
        self, query: str, k: int = 3, *, filters: MetadataFilter | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._chunks:
            return []
        seeds = self._query_seeds(query)
        if not seeds:
            return []
        weights = self.graph.multi_hop_scores(seeds, hops=self.hops, decay=self.decay)
        if not weights:
            return []

        results: list[SearchResult] = []
        for chunk in self._chunks:
            if not matches_filters(chunk, filters):
                continue
            chunk_entities = self.graph.chunk_entities(chunk.id)
            score = sum(
                weights[entity] * self._idf.get(entity, 1.0)
                for entity in chunk_entities
                if entity in weights
            )
            if score > 0.0:
                results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda item: (-item.score, item.chunk.id))
        return results[:k]
