from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterable, Sequence

Vector = list[float]

_sumprod: Callable[[Iterable[float], Iterable[float]], float] | None = getattr(
    math, "sumprod", None
)


def _fast_dot(a: Sequence[float], b: Sequence[float]) -> float:
    if _sumprod is not None:
        return float(_sumprod(a, b))
    return float(sum(map(operator.mul, a, b)))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors have different lengths: {len(a)} and {len(b)}")
    return _fast_dot(a, b)


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(_fast_dot(vector, vector))


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
