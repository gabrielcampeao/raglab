from __future__ import annotations

import math
from collections.abc import Sequence

Vector = list[float]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors have different lengths: {len(a)} and {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalize(vector: Sequence[float]) -> Vector:
    length = norm(vector)
    if length == 0:
        return [0.0] * len(vector)
    return [value / length for value in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    product = dot(a, b)
    lengths = norm(a) * norm(b)
    if lengths == 0:
        return 0.0
    return product / lengths
