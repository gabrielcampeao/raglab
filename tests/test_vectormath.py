import math

import pytest

from raglab.vectormath import cosine_similarity, dot, norm, normalize


def test_dot_product() -> None:
    value = dot([1, 2, 3], [4, 5, 6])
    assert value == 32.0
    assert isinstance(value, float)


def test_dot_rejects_vectors_of_different_lengths() -> None:
    with pytest.raises(ValueError):
        dot([1, 2], [1, 2, 3])


def test_normalize_returns_unit_vector() -> None:
    assert math.isclose(norm(normalize([3.0, 4.0])), 1.0)


def test_normalize_keeps_zero_vector() -> None:
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 0.0], [2.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([0.0, 0.0], [1.0, 1.0], 0.0),
    ],
)
def test_cosine_similarity(a: list[float], b: list[float], expected: float) -> None:
    assert math.isclose(cosine_similarity(a, b), expected, abs_tol=1e-12)
