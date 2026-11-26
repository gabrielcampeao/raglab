import math

import pytest

from raglab.embeddings import HashingEmbedder, TfidfEmbedder, extract_features
from raglab.vectormath import cosine_similarity, norm


def test_extract_features_adds_bigrams_when_requested() -> None:
    assert extract_features("refund policy window", use_bigrams=True) == [
        "refund",
        "policy",
        "window",
        "refund policy",
        "policy window",
    ]


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimensions=64)

    first, second = embedder.embed(["refund policy", "refund policy"])

    assert first == second
    assert len(first) == 64
    assert math.isclose(norm(first), 1.0)


def test_hashing_embedder_places_related_text_closer() -> None:
    embedder = HashingEmbedder()

    query = embedder.embed_query("how do I get a refund")
    related = embedder.embed_query("request a refund within 30 days")
    unrelated = embedder.embed_query("keyboard shortcuts for quick search")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_hashing_embedder_maps_stopwords_only_text_to_zero_vector() -> None:
    assert HashingEmbedder(dimensions=8).embed_query("the and of") == [0.0] * 8


def test_hashing_embedder_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dimensions=0)


def test_tfidf_embedder_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        TfidfEmbedder().embed(["anything"])


def test_tfidf_embedder_weights_rare_terms_higher() -> None:
    embedder = TfidfEmbedder().fit(["refund policy", "refund window", "sync conflict"])

    assert set(embedder.vocabulary) == {"refund", "policy", "window", "sync", "conflict"}
    assert embedder.idf[embedder.vocabulary["policy"]] > embedder.idf[embedder.vocabulary["refund"]]


def test_tfidf_embedder_ignores_unknown_terms() -> None:
    embedder = TfidfEmbedder().fit(["refund policy"])

    assert embedder.embed_query("banana") == [0.0, 0.0]


def test_tfidf_embedder_ranks_matching_document_first() -> None:
    texts = ["refund policy for annual plans", "sync conflict on two devices"]
    embedder = TfidfEmbedder(use_bigrams=True).fit(texts)
    documents = embedder.embed(texts)
    query = embedder.embed_query("annual refund")

    assert cosine_similarity(query, documents[0]) > cosine_similarity(query, documents[1])
