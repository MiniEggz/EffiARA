import numpy as np
import pandas as pd
import pytest

from effiara.agreement import pairwise_cosine_similarity


# valid cases
@pytest.mark.parametrize(
    "data, expected_output",
    [
        # example with two perfectly matching soft label distributions (cosine similarity = 1)
        (
            {
                "col1": [
                    np.array([0.5, 0.5]),
                    np.array([0.1, 0.9]),
                    np.array([0.3, 0.7]),
                ],
                "col2": [
                    np.array([0.5, 0.5]),
                    np.array([0.1, 0.9]),
                    np.array([0.3, 0.7]),
                ],
            },
            pytest.approx(1.0, rel=1e-10),
        ),
        # example where cosine similarity varies
        (
            {
                "col1": [
                    np.array([0.6, 0.4]),
                    np.array([0.2, 0.8]),
                    np.array([0.3, 0.7]),
                ],
                "col2": [
                    np.array([0.5, 0.5]),
                    np.array([0.1, 0.9]),
                    np.array([0.4, 0.6]),
                ],
            },
            pytest.approx(0.98495167, rel=1e-4),
        ),  # Expected mean cosine similarity
        # example where one pair is completely dissimilar
        (
            {
                "col1": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
                "col2": [np.array([0.0, 1.0]), np.array([1.0, 0.0])],
            },
            pytest.approx(0.0, rel=1e-4),
        ),  # Average cosine similarity should be 0
    ],
)
def test_valid_pairwise_cosine_similarity(data, expected_output):
    df = pd.DataFrame(data)
    result = pairwise_cosine_similarity(df, "col1", "col2", num_classes=2)
    assert result == expected_output, f"Expected {expected_output}, got {result}"


# one or both columns contain nans
@pytest.mark.parametrize(
    "data",
    [
        # nan in one column
        {
            "col1": [np.array([0.5, 0.5]), None, np.array([0.3, 0.7])],
            "col2": [np.array([0.5, 0.5]), np.array([0.1, 0.9]), np.array([0.3, 0.7])],
        },
        # nan in both columns
        {
            "col1": [np.array([0.5, 0.5]), None, np.array([0.3, 0.7])],
            "col2": [None, np.array([0.1, 0.9]), np.array([0.3, 0.7])],
        },
    ],
)
def test_nan_values_raise_value_error(data):
    df = pd.DataFrame(data)
    with pytest.raises(
        ValueError, match="One or both of the columns given contain NaN values"
    ):
        pairwise_cosine_similarity(df, "col1", "col2", num_classes=2)


# columns don't contain probabilistic values
@pytest.mark.parametrize(
    "data",
    [
        # non-numeric values
        {
            "col1": ["text", np.array([0.5, 0.5]), np.array([0.3, 0.7])],
            "col2": [np.array([0.5, 0.5]), "text", np.array([0.3, 0.7])],
        },
        # prob label containing negative
        {
            "col1": ["text", np.array([1.5, -0.5]), np.array([0.3, 0.7])],
            "col2": [np.array([0.5, 0.5]), "text", np.array([0.3, 0.7])],
        },
        # lists instead of np.arrays
        {
            "col1": [
                [0.5, 0.5],
                [0.2, 0.8],
                [0.3, 0.7],
            ],  # Lists instead of numpy arrays
            "col2": [[0.5, 0.5], [0.1, 0.9], [0.4, 0.6]],
        },
        # incorrect probability distribution (does not sum to 1)
        {
            "col1": [np.array([0.6, 0.6]), np.array([0.2, 0.8]), np.array([0.3, 0.7])],
            "col2": [np.array([0.5, 0.5]), np.array([0.1, 0.9]), np.array([0.4, 0.6])],
        },
    ],
)
def test_non_probabilistic_labels_raise_exception(data):
    df = pd.DataFrame(data)
    with pytest.raises(Exception, match="No probabilistic labels found in dataframe."):
        pairwise_cosine_similarity(df, "col1", "col2", num_classes=2)


# empty dataframe
def test_empty_dataframe():
    df = pd.DataFrame(columns=["col1", "col2"])  # no rows
    with pytest.raises(ValueError, match="One or both of the columns is empty."):
        pairwise_cosine_similarity(df, "col1", "col2", num_classes=2)
