import numpy as np
import pandas as pd
import pytest

from effiara.data_generator import (
    annotate_samples,
    concat_annotations,
    consolidate_reannotation,
    generate_annotator_label,
    generate_data,
    generate_samples,
    user_df_merge,
)
from effiara.preparation import SampleDistributor


# test generate_annotator_label function
@pytest.mark.parametrize(
    "true_label, correctness_prob, num_classes",
    [
        (0, 1.0, 5),  # Always correct
        (1, 0.0, 5),  # Always wrong, different from true label
        (2, 0.5, 5),  # 50% chance of being correct or incorrect
    ],
)
def test_generate_annotator_label(true_label, correctness_prob, num_classes):
    if correctness_prob == 0.0:
        assert (
            generate_annotator_label(true_label, correctness_prob, num_classes)
            != true_label
        )
    elif correctness_prob == 1.0:
        assert (
            generate_annotator_label(true_label, correctness_prob, num_classes)
            == true_label
        )
    else:
        assert generate_annotator_label(
            true_label, correctness_prob, num_classes
        ) in range(num_classes)


# test generate samples function
def test_generate_samples():
    df = generate_samples(10, 3, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "true_label" in df.columns
    assert df["true_label"].notna().prod() == 1


# test incorrect user input
def test_annotate_samples_invalid_users():
    user_df_dict = {"user_1": pd.DataFrame({"true_label": [0, 1]})}
    user_correctness = {"user_2": 0.8}  # Different user keys
    with pytest.raises(
        ValueError, match="Found different users in user_df_dict and user_correctness"
    ):
        annotate_samples(user_df_dict, user_correctness, 3)


# test annotate samples function
def test_annotate_samples():
    user_df_dict = {
        "user_1": pd.DataFrame({"true_label": [0, 1], "is_reannotation": [False, True]})
    }
    user_correctness = {"user_1": 0.8}
    result = annotate_samples(user_df_dict, user_correctness, 3)
    assert "user_1_label" in result["user_1"].columns
    # only one should be added to regular annotaion column
    assert result["user_1"]["user_1_label"].notna().sum() == 1
    assert "re_user_1_label" in result["user_1"].columns
    # one should be added to reannotation column
    assert result["user_1"]["re_user_1_label"].notna().sum() == 1


# test consolidate_reannotation function
def test_consolidate_reannotation():
    group = pd.DataFrame(
        {
            "sample_id": [1, 1],
            "user_1_label": [1, None],
            "re_user_1_label": [None, 2],
            "is_reannotation": [False, True],
        }
    )
    result = consolidate_reannotation(group)
    assert isinstance(result, pd.Series)
    assert result["sample_id"] == 1
    assert result["user_1_label"] == 1
    assert result["re_user_1_label"] == 2


# test user_df_merge function
def test_user_df_merge():
    df1 = pd.DataFrame({"sample_id": [1, 2], "user_1_label": [0, 1]})
    df2 = pd.DataFrame({"sample_id": [1, 3], "user_2_label": [1, 2]})
    result = user_df_merge(df1, df2)
    assert len(result) == 3
    assert "user_1_label" in result.columns
    assert result["user_1_label"].notna().sum() == 2
    assert "user_2_label" in result.columns
    assert result["user_2_label"].notna().sum() == 2


# test inconsistent values in user_df_merge
def test_user_df_merge_inconsistent_values():
    df1 = pd.DataFrame({"sample_id": [1], "user_1_label": [0]})
    df2 = pd.DataFrame({"sample_id": [1], "user_1_label": [1]})
    with pytest.raises(ValueError, match="Inconsistent values in user_1_label"):
        user_df_merge(df1, df2)


# test concat_annotations function
def test_concat_annotations():
    annotations_dict = {
        "user_1": pd.DataFrame({"sample_id": [1, 2], "user_1_label": [0, 1]}),
        "user_2": pd.DataFrame({"sample_id": [1, 3], "user_2_label": [1, 2]}),
    }
    result = concat_annotations(annotations_dict)
    assert len(result) == 3
    assert "user_1_label" in result.columns
    assert result["user_1_label"].notna().sum() == 2
    assert "user_2_label" in result.columns
    assert result["user_2_label"].notna().sum() == 2


# TODO: update the expected behaviour of small number of annotators
# test generate_data function
def test_generate_data():
    sample_distributor = SampleDistributor(
        annotators=[f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=None,  # 2160
        double_proportion=1.0 / 3,
        re_proportion=0.5,
    )

    sample_distributor.set_project_distribution()
    print(sample_distributor.single_annotation_project)
    print(sample_distributor.double_annotation_project)
    print(sample_distributor.re_annotation_project)
    print(sample_distributor.num_samples)

    annotator_dict = {
        "user_1": 0.8,
        "user_2": 0.7,
        "user_3": 0.4,
        "user_4": 0.9,
        "user_5": 0.95,
        "user_6": 0.75,
    }
    result = generate_data(sample_distributor, annotator_dict, 3)
    assert isinstance(result, pd.DataFrame)
    assert "true_label" in result.columns
    for i in range(1, 7):
        assert f"user_{i}_label" in result.columns
        assert result[f"user_{i}_label"].notna().sum() == 480
