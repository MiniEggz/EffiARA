import numpy as np
import pytest

from effiara.utils import is_prob_label


# testing is_prob_label
# valid
@pytest.mark.parametrize(
    "x, num_classes",
    [
        (np.array([0.2, 0.3, 0.5]), 3),
        (np.array([1.0, 0.0, 0.0]), 3),
        (np.array([0.1, 0.9]), 2),
        (np.array([0.33, 0.33, 0.34]), 3),
        (np.array([0.25, 0.25, 0.25, 0.25]), 4),
    ],
)
def test_valid_prob_labels(x, num_classes):
    assert is_prob_label(
        x, num_classes
    ), f"Expected {x} to be a valid probability label"


# invalid
@pytest.mark.parametrize(
    "x, num_classes",
    [
        (np.array([0.2, 0.3]), 3),  # incorrect shape
        (np.array([0.1, 0.2, 0.3, 0.4]), 3),  # more elements than num_classes
        ([0.2, 0.3, 0.5], 3),  # not a numpy array (list instead)
        ((0.2, 0.3, 0.5), 3),  # not a numpy array (tuple instead)
        (np.array([-0.5, 0.5, 1]), 3),  # negative value
        (np.array([2, -0.5, -0.5]), 3),  # val over one and negative vals
        (np.array([2, 0, 0]), 3),  # val over one
        (np.array([[0.2, 0.3, 0.5]]), 3),  # 2D array instead of 1D
        (np.array(0.5), 1),  # scalar value instead of an array
        ("not an array", 3),  # string input
        (None, 3),  # None input
    ],
)
def test_invalid_prob_labels(x, num_classes):
    assert not is_prob_label(
        x, num_classes
    ), f"Expected {x} to be an invalid probability label"


# invalid edge cases
@pytest.mark.parametrize(
    "x, num_classes",
    [
        (np.array([]), 3),  # empty array
        (np.array([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]]), 3),  # 2D array (invalid)
        (np.array([[0.7], [0.2], [0.1]]), 3),
    ],
)  # maybe more edge cases with shape
def test_edge_cases(x, num_classes):
    assert not is_prob_label(x, num_classes), f"Edge case failed for input {x}"
