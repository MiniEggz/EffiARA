import pandas as pd
import pytest

from effiara.preparation import (
    sample_without_replacement,
)


# test valid cases
@pytest.mark.parametrize(
    "data, n",
    [
        # normal case: Sampling 2 from 5 rows
        ({"col1": [1, 2, 3, 4, 5]}, 2),
        # sampling all rows
        ({"col1": ["a", "b", "c", "d"]}, 4),
        # sampling only 1 row
        ({"col1": [10, 20, 30]}, 1),
    ],
)
def test_valid_sample_without_replacement(data, n):
    df = pd.DataFrame(data)
    remaining_df, sampled_df = sample_without_replacement(df, n)

    assert isinstance(
        remaining_df, pd.DataFrame
    ), "Remaining dataset should be a DataFrame"
    assert isinstance(sampled_df, pd.DataFrame), "Sampled dataset should be a DataFrame"
    assert len(sampled_df) == n, f"Expected {n} sampled rows, got {len(sampled_df)}"
    assert (
        len(remaining_df) == len(df) - n
    ), f"Expected {len(df) - n} remaining rows, got {len(remaining_df)}"
    assert sampled_df.index.isin(
        df.index
    ).all(), "Sampled indices should come from original DataFrame"
    assert (
        len(set(sampled_df.index).intersection(set(remaining_df.index))) == 0
    ), "No overlap between sampled and remaining data"


# takes rest when n > len(df)
@pytest.mark.parametrize(
    "data, n",
    [
        ({"col1": [1, 2, 3]}, 10),  # requesting more than available
        ({"col1": [1]}, 5),  # only one row available
    ],
)
def test_sample_more_than_available(data, n):
    df = pd.DataFrame(data)
    remaining_df, sampled_df = sample_without_replacement(df, n)

    assert len(sampled_df) == len(df), "Should have sampled all available rows"
    assert (
        remaining_df.empty
    ), "Remaining DataFrame should be empty when sampling everything"


# TODO: update to be zero or less
# test when trying to sample zero rows
@pytest.mark.parametrize(
    "data, n",
    [
        ({"col1": [1, 2, 3]}, 0),  # request zero
        ({"col1": [1, 2, 3]}, -5),  # request less than zero
    ],
)
def test_sample_zero_rows(data, n):
    df = pd.DataFrame(data)

    with pytest.raises(
        ValueError, match="Should not attempt to sample 0 or less from DataFrame."
    ):
        remaining_df, sampled_df = sample_without_replacement(df, n)


# sampling empty dataframe should return two empty dataframes
def test_sample_empty_dataframe():
    df = pd.DataFrame(columns=["col1"])
    remaining_df, sampled_df = sample_without_replacement(df, 2)
    assert remaining_df.empty
    assert sampled_df.empty


# test invalid inputs are rejected
@pytest.mark.parametrize("invalid_n", [-1, "string", 3.5, None])
def test_invalid_n_values(invalid_n):
    df = pd.DataFrame({"col1": [1, 2, 3]})

    with pytest.raises(
        (TypeError, ValueError)
    ):  # Expecting an error for invalid n values
        sample_without_replacement(df, invalid_n)
