from pathlib import Path

import pytest

from raglab.models import Chunk
from raglab.vectorstore import InMemoryVectorStore


def _chunk(index: int) -> Chunk:
    return Chunk(
        id=f"doc#{index}",
        text=f"text {index}",
        document_id="doc",
        position=index,
        metadata={"source": "doc"},
    )


@pytest.fixture
def store() -> InMemoryVectorStore:
    vector_store = InMemoryVectorStore()
    vector_store.add([_chunk(0), _chunk(1), _chunk(2)], [[1, 0], [0, 1], [0.7, 0.7]])
    return vector_store


def test_search_returns_best_matches_first(store: InMemoryVectorStore) -> None:
    results = store.search([1.0, 0.0], k=2)

    assert [result.chunk.id for result in results] == ["doc#0", "doc#2"]
    assert results[0].score > results[1].score


def test_store_reports_size_and_dimensions(store: InMemoryVectorStore) -> None:
    assert len(store) == 3
    assert store.dimensions == 2
    assert [chunk.id for chunk in store.chunks] == ["doc#0", "doc#1", "doc#2"]


def test_add_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore().add([_chunk(0)], [[1.0], [2.0]])


def test_add_rejects_inconsistent_dimensions(store: InMemoryVectorStore) -> None:
    with pytest.raises(ValueError):
        store.add([_chunk(3)], [[1.0, 2.0, 3.0]])


def test_search_rejects_wrong_query_dimensions(store: InMemoryVectorStore) -> None:
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0])


def test_search_rejects_non_positive_k(store: InMemoryVectorStore) -> None:
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=0)


def test_search_on_empty_store_returns_nothing() -> None:
    assert InMemoryVectorStore().search([1.0]) == []


def test_clear_removes_everything(store: InMemoryVectorStore) -> None:
    store.clear()

    assert len(store) == 0
    assert store.dimensions is None


def test_save_and_load_round_trip(store: InMemoryVectorStore, tmp_path: Path) -> None:
    path = tmp_path / "index.json"

    store.save(path)
    restored = InMemoryVectorStore.load(path)

    assert restored.chunks == store.chunks
    assert restored.search([0.0, 1.0], k=1)[0].chunk.id == "doc#1"
