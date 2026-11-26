from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from raglab.text import tokenize
from raglab.vectormath import Vector, normalize


def extract_features(text: str, *, use_bigrams: bool) -> list[str]:
    tokens = tokenize(text)
    if not use_bigrams:
        return tokens
    return tokens + [f"{first} {second}" for first, second in pairwise(tokens)]


def _log_weight(count: int) -> float:
    return 1.0 + math.log(count)


class Embedder(ABC):
    def fit(self, texts: Sequence[str]) -> Embedder:
        return self

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        raise NotImplementedError

    def embed_query(self, text: str) -> Vector:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    def __init__(self, dimensions: int = 1024, *, use_bigrams: bool = True) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.use_bigrams = use_bigrams

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value >> 63 == 0 else -1.0
        return value % self.dimensions, sign

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            counts = Counter(extract_features(text, use_bigrams=self.use_bigrams))
            for feature, count in counts.items():
                index, sign = self._bucket(feature)
                vector[index] += sign * _log_weight(count)
            vectors.append(normalize(vector))
        return vectors


class TfidfEmbedder(Embedder):
    def __init__(self, *, use_bigrams: bool = False) -> None:
        self.use_bigrams = use_bigrams
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> TfidfEmbedder:
        document_features = [
            set(extract_features(text, use_bigrams=self.use_bigrams)) for text in texts
        ]
        frequencies = Counter(feature for features in document_features for feature in features)
        terms = sorted(frequencies)
        total = len(document_features)
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        self.idf = [math.log((1 + total) / (1 + frequencies[term])) + 1.0 for term in terms]
        self._fitted = True
        return self

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fitted before it can embed text")
        vectors: list[Vector] = []
        for text in texts:
            vector = [0.0] * len(self.vocabulary)
            counts = Counter(extract_features(text, use_bigrams=self.use_bigrams))
            for feature, count in counts.items():
                index = self.vocabulary.get(feature)
                if index is not None:
                    vector[index] = _log_weight(count) * self.idf[index]
            vectors.append(normalize(vector))
        return vectors
