import numpy as np
import pandas as pd
import pytest

from effiara.agreement import calculate_krippendorff_alpha_per_label


# valid calculations
@pytest.mark.parametrize(
    "data, expected_output",
    [
        # perfect agreement (krippendorff's alpha = 1)
        (
            {
                "annotator_1": [
                    np.array([1, 0, 1]),
                    np.array([0, 1, 0]),
                    np.array([1, 1, 1]),
                ],
                "annotator_2": [
                    np.array([1, 0, 1]),
                    np.array([0, 1, 0]),
                    np.array([1, 1, 1]),
                ],
            },
            1.0,
        ),
        # partial agreement (krippendorff's alpha between 0 and 1)
        (
            {
                "annotator_1": [
                    np.array([1, 0, 1]),
                    np.array([0, 1, 0]),
                    np.array([1, 1, 1]),
                ],
                "annotator_2": [
                    np.array([0, 1, 1]),
                    np.array([1, 0, 0]),
                    np.array([1, 0, 1]),
                ],
            },
            pytest.approx(0.027777, rel=1e-4),
        ),
        # no agreement (krippendorff's alpha negative)
        (
            {
                "annotator_1": [
                    np.array([1, 0, 1]),
                    np.array([0, 1, 0]),
                    np.array([1, 1, 1]),
                ],
                "annotator_2": [
                    np.array([0, 1, 0]),
                    np.array([1, 0, 1]),
                    np.array([0, 0, 0]),
                ],
            },
            pytest.approx(-0.66666, rel=1e-4),
        ),
        # lists instead of np.arrays
        (
            {
                "annotator_1": [
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 1, 1],
                ],  # Lists instead of numpy arrays
                "annotator_2": [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
            },
            pytest.approx(0.027777, rel=1e-4),
        ),
    ],
)
def test_valid_krippendorff_alpha_per_label(data, expected_output):
    df = pd.DataFrame(data)
    result = calculate_krippendorff_alpha_per_label(
        df, "annotator_1", "annotator_2", agreement_type="nominal"
    )
    assert result == expected_output, f"Expected {expected_output}, got {result}"


# matrices not the same shape
@pytest.mark.parametrize(
    "data",
    [
        # different lengths
        {
            "annotator_1": [np.array([1, 0]), np.array([0, 1, 0]), np.array([1, 1, 1])],
            "annotator_2": [np.array([1, 0, 1]), np.array([0, 1, 0]), np.array([1, 1])],
        },
        # different number of labels per annotation
        {
            "annotator_1": [
                np.array([1, 0, 1, 0]),
                np.array([0, 1, 0, 0]),
                np.array([1, 1, 1, 1]),
            ],
            "annotator_2": [
                np.array([1, 0, 1]),
                np.array([0, 1, 0]),
                np.array([1, 1, 1]),
            ],
        },
    ],
)
def test_mismatched_annotation_shapes(data):
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        calculate_krippendorff_alpha_per_label(
            df, "annotator_1", "annotator_2", agreement_type="nominal"
        )


# incorrect data formats
@pytest.mark.parametrize(
    "data",
    [
        # non-numeric annotations
        {"annotator_1": ["A", "B", "C"], "annotator_2": ["A", "B", "C"]},
        # mixed types in columns
        {
            "annotator_1": [np.array([1, 0, 1]), "text", np.array([1, 1, 1])],
            "annotator_2": [
                np.array([1, 0, 1]),
                np.array([0, 1, 0]),
                np.array([1, 1, 1]),
            ],
        },
    ],
)
def test_invalid_data_formats(data):
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        calculate_krippendorff_alpha_per_label(
            df, "annotator_1", "annotator_2", agreement_type="nominal"
        )


# empty dataframe
def test_empty_dataframe():
    df = pd.DataFrame(columns=["annotator_1", "annotator_2"])  # No rows
    with pytest.raises(ValueError):
        calculate_krippendorff_alpha_per_label(
            df, "annotator_1", "annotator_2", agreement_type="nominal"
        )


# TODO: might be worth adding some kind
# of acceptance tests with ordinal and
# nominal data.
