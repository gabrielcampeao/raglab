from __future__ import annotations

import heapq
import math
import random
from collections.abc import Iterable, Mapping, Sequence

from raglab.models import Chunk, SearchResult
from raglab.vectormath import Vector, norm, normalize
from raglab.vectorstore import VectorStore, _matches_metadata


def _cosine_distance(first: Sequence[float], second: Sequence[float]) -> float:
    dot_product = sum(x * y for x, y in zip(first, second, strict=True))
    return 1.0 - max(-1.0, min(1.0, dot_product))


class HNSWVectorStore(VectorStore):
    def __init__(
        self,
        *,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 48,
        seed: int = 42,
    ) -> None:
        if m < 2:
            raise ValueError("m must be at least 2")
        if ef_construction <= 0:
            raise ValueError("ef_construction must be positive")
        if ef_search <= 0:
            raise ValueError("ef_search must be positive")
        self.m = m
        self.m_max0 = m * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._seed = seed
        self._level_mult = 1.0 / math.log(m)
        self._rng = random.Random(seed)
        self._chunks: list[Chunk] = []
        self._vectors: list[Vector] = []
        self._neighbors: list[list[list[int]]] = []
        self._entry_point: int | None = None
        self._max_level = -1

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimensions(self) -> int | None:
        return len(self._vectors[0]) if self._vectors else None

    @property
    def max_level(self) -> int:
        return self._max_level

    def _sample_level(self) -> int:
        draw = max(self._rng.random(), 1e-12)
        return int(-math.log(draw) * self._level_mult)

    def _greedy_descent(self, query: Sequence[float], entry: int, layer: int) -> tuple[int, float]:
        current = entry
        current_dist = _cosine_distance(query, self._vectors[current])
        improved = True
        while improved:
            improved = False
            for neighbor in self._neighbors[current][layer]:
                dist = _cosine_distance(query, self._vectors[neighbor])
                if dist < current_dist:
                    current = neighbor
                    current_dist = dist
                    improved = True
        return current, current_dist

    def _search_layer(
        self, query: Sequence[float], entry: int, ef: int, layer: int
    ) -> list[tuple[float, int]]:
        entry_dist = _cosine_distance(query, self._vectors[entry])
        visited = {entry}
        candidates: list[tuple[float, int]] = [(entry_dist, entry)]
        best: list[tuple[float, int]] = [(-entry_dist, entry)]

        while candidates:
            cand_dist, cand_id = heapq.heappop(candidates)
            worst_dist = -best[0][0]
            if cand_dist > worst_dist and len(best) >= ef:
                break
            for neighbor in self._neighbors[cand_id][layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                dist = _cosine_distance(query, self._vectors[neighbor])
                if len(best) < ef or dist < -best[0][0]:
                    heapq.heappush(candidates, (dist, neighbor))
                    heapq.heappush(best, (-dist, neighbor))
                    if len(best) > ef:
                        heapq.heappop(best)

        ranked = [(-neg_dist, node_id) for neg_dist, node_id in best]
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        return ranked

    def _select_neighbors(self, candidates: list[tuple[float, int]], limit: int) -> list[int]:
        if len(candidates) <= limit:
            return [node_id for _, node_id in candidates]
        selected: list[int] = []
        discarded: list[int] = []
        for dist_to_query, node_id in candidates:
            node_vec = self._vectors[node_id]
            closer_to_selected = any(
                _cosine_distance(node_vec, self._vectors[chosen]) < dist_to_query
                for chosen in selected
            )
            if not closer_to_selected:
                selected.append(node_id)
                if len(selected) == limit:
                    return selected
            else:
                discarded.append(node_id)
        for node_id in discarded:
            selected.append(node_id)
            if len(selected) == limit:
                break
        return selected

    def _insert_node(self, chunk: Chunk, vector: Vector) -> None:
        node_id = len(self._chunks)
        level = self._sample_level()
        self._chunks.append(chunk)
        self._vectors.append(vector)
        self._neighbors.append([[] for _ in range(level + 1)])

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = level
            return

        entry = self._entry_point
        for layer in range(self._max_level, level, -1):
            entry, _ = self._greedy_descent(vector, entry, layer)

        top_insert = min(level, self._max_level)
        for layer in range(top_insert, -1, -1):
            found = self._search_layer(vector, entry, self.ef_construction, layer)
            if found:
                entry = found[0][1]
            max_conn = self.m_max0 if layer == 0 else self.m
            chosen = self._select_neighbors(found, max_conn)
            self._neighbors[node_id][layer] = list(chosen)
            for peer in chosen:
                peer_links = self._neighbors[peer][layer]
                peer_links.append(node_id)
                if len(peer_links) > max_conn:
                    peer_vec = self._vectors[peer]
                    scored = sorted(
                        (_cosine_distance(peer_vec, self._vectors[other]), other)
                        for other in peer_links
                    )
                    self._neighbors[peer][layer] = self._select_neighbors(scored, max_conn)

        if level > self._max_level:
            self._entry_point = node_id
            self._max_level = level

    def add(self, chunks: Iterable[Chunk], vectors: Iterable[Sequence[float]]) -> None:
        new_chunks = list(chunks)
        raw_vectors = [[float(value) for value in vector] for vector in vectors]
        if len(new_chunks) != len(raw_vectors):
            raise ValueError("Each chunk needs exactly one vector")
        if not raw_vectors:
            return
        expected = self.dimensions if self.dimensions is not None else len(raw_vectors[0])
        if expected == 0:
            raise ValueError("Vectors must have at least one dimension")
        if any(len(vector) != expected for vector in raw_vectors):
            raise ValueError(f"All vectors must have {expected} dimensions")
        for chunk, raw in zip(new_chunks, raw_vectors, strict=True):
            self._insert_node(chunk, normalize(raw))

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._neighbors.clear()
        self._entry_point = None
        self._max_level = -1
        self._rng = random.Random(self._seed)

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 3,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if self._entry_point is None:
            return []
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query has {len(query_vector)} dimensions but the store holds {self.dimensions}"
            )
        if norm(query_vector) == 0.0:
            matching = [
                SearchResult(chunk=chunk, score=0.0)
                for chunk in self._chunks
                if _matches_metadata(chunk, filters)
            ]
            matching.sort(key=lambda item: item.chunk.id)
            return matching[:k]

        query = normalize(query_vector)
        entry = self._entry_point
        for layer in range(self._max_level, 0, -1):
            entry, _ = self._greedy_descent(query, entry, layer)

        ef = max(self.ef_search, k * 4 if filters else k)
        beam = self._search_layer(query, entry, ef, 0)
        results = [
            SearchResult(chunk=self._chunks[node_id], score=1.0 - dist)
            for dist, node_id in beam
            if _matches_metadata(self._chunks[node_id], filters)
        ]

        if filters and len(results) < k:
            exact_filtered = [
                SearchResult(
                    chunk=chunk,
                    score=1.0 - _cosine_distance(query, vector),
                )
                for chunk, vector in zip(self._chunks, self._vectors, strict=True)
                if _matches_metadata(chunk, filters)
            ]
            exact_filtered.sort(key=lambda item: (-item.score, item.chunk.id))
            return exact_filtered[:k]

        results.sort(key=lambda item: (-item.score, item.chunk.id))
        return results[:k]
