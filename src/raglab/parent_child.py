from __future__ import annotations

from collections.abc import Iterable

from raglab.chunking import chunk_documents, chunk_markdown_documents, chunk_text
from raglab.models import Chunk, Document, SearchResult
from raglab.retrievers import MetadataFilter, Retriever


def _validate_hierarchy(parent_max_chars: int, child_max_chars: int) -> None:
    if parent_max_chars <= 0 or child_max_chars <= 0:
        raise ValueError("parent_max_chars and child_max_chars must be positive")
    if child_max_chars > parent_max_chars:
        raise ValueError("child_max_chars cannot exceed parent_max_chars")


def _split_parent_into_children(parent: Chunk, child_max_chars: int) -> list[Chunk]:
    pieces = chunk_text(parent.text, max_chars=child_max_chars, overlap=0)
    if not pieces:
        return []
    children: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        child_meta = {**parent.metadata, "parent_id": parent.id}
        children.append(
            Chunk(
                id=f"{parent.id}::c{idx}",
                text=piece,
                document_id=parent.document_id,
                position=idx,
                metadata=child_meta,
            )
        )
    return children


def build_parent_child_chunks(
    documents: Iterable[Document],
    *,
    parent_max_chars: int = 750,
    child_max_chars: int = 180,
    parent_overlap: int = 0,
    markdown_aware: bool = True,
) -> tuple[list[Chunk], list[Chunk]]:
    _validate_hierarchy(parent_max_chars, child_max_chars)
    chunker = chunk_markdown_documents if markdown_aware else chunk_documents
    parents = chunker(documents, max_chars=parent_max_chars, overlap=parent_overlap)
    children: list[Chunk] = []
    for parent in parents:
        children.extend(_split_parent_into_children(parent, child_max_chars))
    return parents, children


class ParentChildRetriever(Retriever):
    def __init__(
        self,
        retriever: Retriever,
        *,
        parent_max_chars: int = 750,
        child_max_chars: int = 180,
        markdown_aware: bool = True,
        child_pool_multiplier: int = 4,
    ) -> None:
        _validate_hierarchy(parent_max_chars, child_max_chars)
        if child_pool_multiplier <= 0:
            raise ValueError("child_pool_multiplier must be positive")
        self.retriever = retriever
        self.parent_max_chars = parent_max_chars
        self.child_max_chars = child_max_chars
        self.markdown_aware = markdown_aware
        self.child_pool_multiplier = child_pool_multiplier
        self._parents: dict[str, Chunk] = {}

    def chunk_parents(self, documents: Iterable[Document]) -> list[Chunk]:
        chunker = chunk_markdown_documents if self.markdown_aware else chunk_documents
        return chunker(documents, max_chars=self.parent_max_chars, overlap=0)

    def ingest_documents(self, documents: Iterable[Document]) -> tuple[int, int]:
        parents, children = build_parent_child_chunks(
            documents,
            parent_max_chars=self.parent_max_chars,
            child_max_chars=self.child_max_chars,
            markdown_aware=self.markdown_aware,
        )
        self._parents = {parent.id: parent for parent in parents}
        self.retriever.index(children)
        return len(parents), len(children)

    def index(self, chunks: Iterable[Chunk]) -> None:
        parent_list = list(chunks)
        self._parents = {parent.id: parent for parent in parent_list}
        children: list[Chunk] = []
        for parent in parent_list:
            children.extend(_split_parent_into_children(parent, self.child_max_chars))
        self.retriever.index(children)

    def retrieve(
        self, query: str, k: int = 3, *, filters: MetadataFilter | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._parents:
            return []
        pool = max(k * self.child_pool_multiplier, 16)
        child_hits = (
            self.retriever.retrieve(query, pool, filters=filters)
            if filters is not None
            else self.retriever.retrieve(query, pool)
        )
        best_scores: dict[str, float] = {}
        for hit in child_hits:
            parent_id = hit.chunk.metadata.get("parent_id", hit.chunk.id)
            if parent_id not in self._parents:
                continue
            current = best_scores.get(parent_id)
            if current is None or hit.score > current:
                best_scores[parent_id] = hit.score
        results = [
            SearchResult(chunk=self._parents[pid], score=score)
            for pid, score in best_scores.items()
        ]
        results.sort(key=lambda item: (-item.score, item.chunk.id))
        return results[:k]
