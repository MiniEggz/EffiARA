import pandas as pd
import pytest

from effiara.agreement import pairwise_cohens_kappa_agreement


# valid agreement calculations
@pytest.mark.parametrize(
    "data, label_mapping, expected_output",
    [
        # perfect agreement (cohen's kappa = 1)
        (
            {
                "annotator_1": ["A", "B", "C", "A", "B"],
                "annotator_2": ["A", "B", "C", "A", "B"],
            },
            {"A": 0, "B": 1, "C": 2},
            1.0,
        ),
        # partial agreement (cohen's kappa between 0 and 1)
        (
            {
                "annotator_1": ["A", "A", "C", "B", "B"],
                "annotator_2": ["A", "B", "C", "A", "B"],
            },
            {"A": 0, "B": 1, "C": 2},
            pytest.approx(0.375, rel=1e-4),
        ),
        # no agreement at all (cohen's kappa negative)
        (
            {
                "annotator_1": ["A", "B", "C", "A", "B"],
                "annotator_2": ["B", "C", "A", "B", "C"],
            },
            {"A": 0, "B": 1, "C": 2},
            pytest.approx(-0.4705882352, rel=1e-4),
        ),
    ],
)
def test_valid_cohens_kappa_agreement(data, label_mapping, expected_output):
    df = pd.DataFrame(data)
    result = pairwise_cohens_kappa_agreement(
        df, "annotator_1", "annotator_2", label_mapping
    )
    assert result == expected_output, f"Expected {expected_output}, got {result}"


# nan values in one or both columns
@pytest.mark.parametrize(
    "data",
    [
        # nan in one column
        {
            "annotator_1": ["A", None, "C", "A", "B"],
            "annotator_2": ["A", "B", "C", "A", "B"],
        },
        # nan in both columns
        {
            "annotator_1": ["A", None, "C", "A", "B"],
            "annotator_2": ["B", "C", None, "A", "B"],
        },
    ],
)
def test_nan_values_raise_value_error(data):
    label_mapping = {"A": 0, "B": 1, "C": 2}
    df = pd.DataFrame(data)
    with pytest.raises(
        ValueError, match="One or both of the columns given contain NaN values"
    ):
        pairwise_cohens_kappa_agreement(df, "annotator_1", "annotator_2", label_mapping)


# invalid label mappings
@pytest.mark.parametrize(
    "data, label_mapping",
    [
        # missing label in mapping
        (
            {
                "annotator_1": ["A", "B", "C"],
                "annotator_2": ["A", "B", "D"],  # 'D' is not in the label mapping
            },
            {"A": 0, "B": 1, "C": 2},
        ),
        # empty label mapping
        ({"annotator_1": ["A", "B", "C"], "annotator_2": ["A", "B", "C"]}, {}),
    ],
)
def test_invalid_label_mapping(data, label_mapping):
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Unexpected label found;"):
        pairwise_cohens_kappa_agreement(df, "annotator_1", "annotator_2", label_mapping)


# empty dataframe
def test_empty_dataframe():
    df = pd.DataFrame(columns=["annotator_1", "annotator_2"])  # No rows
    label_mapping = {"A": 0, "B": 1, "C": 2}
    with pytest.raises(ValueError, match="One or both of the columns is empty."):
        pairwise_cohens_kappa_agreement(df, "annotator_1", "annotator_2", label_mapping)
