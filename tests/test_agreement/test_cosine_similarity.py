import numpy as np
import pytest

from effiara.agreement import cosine_similarity


# valid calculations
@pytest.mark.parametrize(
    "vector_a, vector_b, expected_output",
    [
        (np.array([1, 0]), np.array([1, 0]), 1.0),  # Identical vectors (max similarity)
        (np.array([1, 0]), np.array([0, 1]), 0.0),  # Perpendicular vectors (cosine 90°)
        (np.array([-1, 0]), np.array([1, 0]), -1.0),  # Opposite vectors (cosine 180°)
        (np.array([3, 4]), np.array([6, 8]), 1.0),  # Collinear vectors
        (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            pytest.approx(0.974632, rel=1e-4),
        ),  # General case
        (
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            pytest.approx(0.974632, rel=1e-4),
        ),  # Small values
    ],
)
def test_valid_cosine_similarity(vector_a, vector_b, expected_output):
    result = cosine_similarity(vector_a, vector_b)
    assert result == expected_output, f"Expected {expected_output}, got {result}"


# zero vectors
@pytest.mark.parametrize(
    "vector_a, vector_b",
    [
        (
            np.array([0, 0]),
            np.array([1, 1]),
        ),  # Zero vector should cause division by zero
        (
            np.array([1, 1]),
            np.array([0, 0]),
        ),  # Zero vector should cause division by zero
        (np.array([0, 0, 0]), np.array([0, 0, 0])),  # Both vectors are zero
    ],
)
def test_zero_vector_case(vector_a, vector_b):
    with pytest.raises(ZeroDivisionError):
        cosine_similarity(vector_a, vector_b)


# invalid input
@pytest.mark.parametrize(
    "vector_a, vector_b",
    [
        ("not a vector", np.array([1, 2, 3])),  # first argument is a string
        (np.array([1, 2, 3]), "not a vector"),  # second argument is a string
        (None, np.array([1, 2, 3])),  # first argument is none
        (np.array([1, 2, 3]), None),  # second argument is none
        (
            [1, 2, 3],
            np.array([1, 2, 3]),
        ),  # first argument is a list instead of np.ndarray
        (
            np.array([1, 2, 3]),
            [1, 2, 3],
        ),  # second argument is a list instead of np.ndarray
        (np.array([1, 2]), np.array([1, 2, 3])),  # mismatched dimensions
    ],
)
def test_invalid_inputs(vector_a, vector_b):
    with pytest.raises(ValueError):  # catch any type errors or shape mismatches
        cosine_similarity(vector_a, vector_b)
