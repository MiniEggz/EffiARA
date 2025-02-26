import numpy as np
import pandas as pd
import pytest

from effiara.utils import retrieve_pair_annotations


# valid input
@pytest.mark.parametrize(
    "data, user_x, user_y, expected_rows",
    [
        # both users have annotations for some samples
        (
            {"user1_label": [1, None, 2, 3, None], "user2_label": [1, 2, 2, None, 3]},
            "user1",
            "user2",
            2,
        ),
        # all rows have annotations (should return all rows)
        ({"user1_label": [1, 2, 3], "user2_label": [1, 2, 3]}, "user1", "user2", 3),
        # mixed cases where some users have missing annotations
        (
            {
                "user1_label": [None, 2, 3, None, 5],
                "user2_label": [1, None, 3, None, 5],
            },
            "user1",
            "user2",
            2,
        ),
    ],
)
def test_retrieve_pair_annotations_valid_cases(data, user_x, user_y, expected_rows):
    df = pd.DataFrame(data)
    filtered_df = retrieve_pair_annotations(df, user_x, user_y)

    assert isinstance(filtered_df, pd.DataFrame), "Returned value should be a DataFrame"
    assert (
        len(filtered_df) == expected_rows
    ), f"Expected {expected_rows} rows but got {len(filtered_df)}"


# columns missing / empty dataframe
@pytest.mark.parametrize(
    "data, user_x, user_y",
    [
        ({}, "user1", "user2"),  # empty df
        ({"user1_label": [1, 2, 3]}, "user1", "user2"),  # missing user2_label
        ({"user2_label": [1, 2, 3]}, "user1", "user2"),  # missing user1_label
    ],
)
def test_retrieve_pair_annotations_missing_columns(data, user_x, user_y):
    df = pd.DataFrame(data)
    with pytest.raises(KeyError, match=f".*{user_x}_label|{user_y}_label.*"):
        retrieve_pair_annotations(df, user_x, user_y)


# all missing values
@pytest.mark.parametrize(
    "data, user_x, user_y, expected_rows",
    [
        (
            {"user1_label": [None, None, None], "user2_label": [None, None, None]},
            "user1",
            "user2",
            0,
        ),  # all values are None
        (
            {
                "user1_label": [np.nan, np.nan, np.nan],
                "user2_label": [np.nan, np.nan, np.nan],
            },
            "user1",
            "user2",
            0,
        ),  # all values are NaN
        # no valid rows (no row has both labels)
        (
            {"user1_label": [None, None, None], "user2_label": [1, 2, 3]},
            "user1",
            "user2",
            0,
        ),
    ],
)
def test_retrieve_pair_annotations_edge_cases(data, user_x, user_y, expected_rows):
    df = pd.DataFrame(data)
    filtered_df = retrieve_pair_annotations(df, user_x, user_y)

    assert filtered_df.empty, "Expected an empty DataFrame"
    assert (
        len(filtered_df) == expected_rows
    ), f"Expected {expected_rows} rows but got {len(filtered_df)}"
