import pandas as pd
import pytest

from effiara.agreement import pairwise_agreement


# mock return values for agreement calculations
@pytest.fixture
def mock_functions(mocker):
    mocker.patch(
        "effiara.agreement.pairwise_nominal_krippendorff_agreement",
        return_value=0.8,
    )
    mocker.patch(
        "effiara.agreement.pairwise_cohens_kappa_agreement", return_value=0.75
    )
    mocker.patch(
        "effiara.agreement.pairwise_fleiss_kappa_agreement", return_value=0.7
    )
    mocker.patch("effiara.agreement.pairwise_cosine_similarity", return_value=0.85)
    mocker.patch(
        "effiara.agreement.calculate_krippendorff_alpha_per_label", return_value=0.9
    )


# valid inputs
@pytest.mark.parametrize(
    "metric, expected_output",
    [
        ("krippendorff", 0.8),
        ("cohen", 0.75),
        ("fleiss", 0.7),
        ("cosine", 0.85),
        ("multi_krippendorff", 0.9),
    ],
)
def test_valid_pairwise_agreement(mock_functions, metric, expected_output):
    df = pd.DataFrame(
        {"user1_label": ["A", "B", "C", "A"], "user2_label": ["A", "B", "C", "B"]}
    )
    label_mapping = {"A": 0, "B": 1, "C": 2}

    result = pairwise_agreement(
        df, "user1", "user2", label_mapping, num_classes=3, metric=metric
    )
    assert result == expected_output, f"Expected {expected_output}, got {result}"


# test invalid metric
def test_invalid_metric():
    df = pd.DataFrame({"user1_label": ["A", "B", "C"], "user2_label": ["A", "B", "C"]})
    label_mapping = {"A": 0, "B": 1, "C": 2}

    with pytest.raises(
        ValueError, match="The metric invalid_metric was not recognised."
    ):
        pairwise_agreement(
            df, "user1", "user2", label_mapping, num_classes=3, metric="invalid_metric"
        )


# missing annotations should raise key error
@pytest.mark.parametrize(
    "df_data",
    [
        {
            "user1_label": ["A", "B", "C"],
            "user3_label": ["A", "B", "C"],
        },  # user2_label missing
        {"user2_label": ["A", "B", "C"]},  # user1_label missing
    ],
)
def test_missing_annotations(df_data):
    df = pd.DataFrame(df_data)
    label_mapping = {"A": 0, "B": 1, "C": 2}

    with pytest.raises(KeyError):
        pairwise_agreement(
            df, "user1", "user2", label_mapping, num_classes=3, metric="krippendorff"
        )


# empty dataframe should raise value error
@pytest.mark.parametrize(
    "metric", ["krippendorff", "cohen", "fleiss", "cosine", "multi_krippendorff"]
)
def test_empty_dataframe(metric):
    df = pd.DataFrame(columns=["user1_label", "user2_label"])
    label_mapping = {"A": 0, "B": 1, "C": 2}

    with pytest.raises(ValueError):  # handled in agreement library
        pairwise_agreement(
            df, "user1", "user2", label_mapping, num_classes=3, metric=metric
        )
