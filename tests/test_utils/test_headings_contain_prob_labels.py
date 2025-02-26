import numpy as np
import pandas as pd
import pytest

from effiara.utils import headings_contain_prob_labels


# return true
@pytest.mark.parametrize(
    "data, num_classes",
    [
        # both columns contain only valid probability labels
        (
            {
                "col1": [np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.9, 0.0])],
                "col2": [np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])],
            },
            3,
        ),
        (
            {
                "col1": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
                "col2": [np.array([0.5, 0.5]), np.array([0.2, 0.8])],
            },
            2,
        ),
    ],
)
def test_valid_headings_contain_prob_labels(data, num_classes):
    df = pd.DataFrame(data)
    assert headings_contain_prob_labels(
        df, "col1", "col2", num_classes
    ), "Expected valid probability labels"


# return false
@pytest.mark.parametrize(
    "data, num_classes",
    [
        # a single invalid entry (wrong shape)
        (
            {
                "col1": [
                    np.array([0.2, 0.3]),
                    np.array([0.1, 0.9, 0.0]),
                ],  # First entry has 2 elements instead of 3
                "col2": [np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])],
            },
            3,
        ),
        # a single non-numpy array entry
        (
            {
                "col1": [np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.9, 0.0])],
                "col2": [
                    np.array([0.5, 0.3, 0.2]),
                    [0.3, 0.3, 0.4],
                ],  # List instead of numpy array
            },
            3,
        ),
        # string values in one of the columns
        (
            {
                "col1": [np.array([0.2, 0.3, 0.5]), "invalid"],
                "col2": [np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])],
            },
            3,
        ),
        # columns with none values
        (
            {
                "col1": [np.array([0.2, 0.3, 0.5]), None],
                "col2": [np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])],
            },
            3,
        ),
        # 2d arrays instead of 1d vectors
        (
            {
                "col1": [np.array([[0.2, 0.3, 0.5]]), np.array([0.1, 0.9, 0.0])],
                "col2": [np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])],
            },
            3,
        ),
    ],
)
def test_invalid_headings_contain_prob_labels(data, num_classes):
    df = pd.DataFrame(data)
    assert not headings_contain_prob_labels(
        df, "col1", "col2", num_classes
    ), "Expected invalid probability labels"


# raise key error
@pytest.mark.parametrize(
    "data, num_classes",
    [
        # missing one of the required columns
        ({"col1": [np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.9, 0.0])]}, 3),
        # empty dataframe
        ({}, 3),
    ],
)
def test_edge_cases_headings_contain_prob_labels(data, num_classes):
    df = pd.DataFrame(data)
    with pytest.raises(KeyError):
        headings_contain_prob_labels(df, "col1", "col2", num_classes)
