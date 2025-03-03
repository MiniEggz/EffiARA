"""Functions for generating datasets and annotations."""

from functools import reduce
from typing import Dict

import numpy as np
import pandas as pd

from effiara.preparation import SampleDistributor


def generate_annotator_label(
    true_label: int, correctness_probability: float, num_classes: int
) -> int:
    """Generate an annotator label based on the true label and how correct
       the annotator should be.

    Args:
        true_label (int): the true label of the sample.
        correctness_probability (float): the probability that the annotator
            annotates the sample with the correct label.
        num_classes (int): number of classes

    Returns:
        int: label
    """
    if np.random.rand() < correctness_probability:
        return true_label
    else:
        return np.random.choice(list(set(range(num_classes)) - {true_label}))


def generate_samples(num_samples: int, num_classes: int, seed=None) -> pd.DataFrame:
    """Generate a set of anntotations to be tested in annotator
       reliability assessment. Allows control over how good each
       annotator should be, allowing the assessment of the annotation
       framework.

    Args:
        num_samples (int): the number of samples to generate.
        num_classes (int): the number of possible labels.
        seed (int): the random seed. Default None.

    Returns:
        pd.DataFrame: A dataset of samples.
    """
    if seed is not None:
        np.random.seed(seed)
    true_labels = np.random.randint(0, num_classes, size=num_samples)
    dataset = pd.DataFrame({"true_label": true_labels})

    return dataset


def annotate_samples(
    user_df_dict: Dict[str, pd.DataFrame],
    user_correctness: Dict[str, float],
    num_classes: int,
) -> Dict[str, pd.DataFrame]:
    """Generate annotations according to annotator correctness.

    Args:
        user_df_dict (Dict[str, pd.DataFrame]): dict of usernames: examples.
        user_correctness (Dict[str, pd.DataFrame]): dict from
                                                    usernames to correctness.
        num_classes (int): number of classes.

    Returns:
        Dict[str, pd.DataFrame]: The examples with annotations for each user.
    """

    if set(user_df_dict.keys()) != set(user_correctness.keys()):
        raise ValueError(
            f"Found different users in user_df_dict and user_correctness."
        )  # noqa

    new_user_df_dict = {}
    for user, df in user_df_dict.items():
        # annotate samples
        df.loc[~df["is_reannotation"], f"{user}_label"] = df.loc[
            ~df["is_reannotation"], "true_label"
        ].apply(
            generate_annotator_label,
            correctness_probability=user_correctness[user],
            num_classes=num_classes,
        )
        df.loc[~df["is_reannotation"], f"{user}_confidence"] = 5
        df.loc[~df["is_reannotation"], f"{user}_secondary"] = np.nan

        # reannotations
        df.loc[df["is_reannotation"], f"re_{user}_label"] = df.loc[
            df["is_reannotation"], "true_label"
        ].apply(
            generate_annotator_label,
            correctness_probability=user_correctness[user],
            num_classes=num_classes,
        )
        df.loc[df["is_reannotation"], f"re_{user}_confidence"] = 5
        df.loc[df["is_reannotation"], f"re_{user}_secondary"] = np.nan
        new_user_df_dict[user] = df
    return new_user_df_dict


def consolidate_reannotation(group: pd.DataFrame) -> pd.Series:
    """Helper function to merge a group of rows with the same sample_id.

    Args:
        group (pd.DataFrame): group of dataframes to merge.

    Returns:
        pd.Series: merged row.
    """
    # use first row as base
    base_row = group.iloc[0].copy()

    for col in group.columns:
        if col.startswith("re_"):
            if pd.isna(base_row[col]):
                base_row[col] = (
                    group[col].dropna().iloc[0]
                    if not group[col].dropna().empty
                    else np.nan
                )

    return base_row


def user_df_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge left and right dataframes on sample_id with checks for
       consistency in shared columns.

    Args:
        left (pd.DataFrame): left dataframe to merge.
        right (pd.DataFrame): right dataframe to merge into left.

    Returns:
        pd.DataFrame: merged dataframe.
    """
    if "is_reannotation" in left.columns:
        left = left.drop(columns=["is_reannotation"])
    if "is_reannotation" in right.columns:
        right = right.drop(columns=["is_reannotation"])
    merged = left.merge(right, on="sample_id", how="outer",
                        suffixes=("", "_dup"))

    # loop over duplicate columns
    for col in merged.columns:
        if col.endswith("_dup"):
            original_col = col.replace("_dup", "")
            if original_col in merged.columns:
                merged[original_col] = merged[original_col].combine_first(
                    merged[col]
                )  # noqa

                # check for consistency
                inconsistent = (
                    (merged[original_col] != merged[col])
                    & merged[original_col].notna()
                    & merged[col].notna()
                )
                if inconsistent.any():
                    raise ValueError(f"Inconsistent values in {original_col}.")

            merged.drop(columns=col, inplace=True)

    merged = (
        merged.groupby("sample_id")
        .apply(lambda group: consolidate_reannotation(group))
        .reset_index(drop=True)
    )

    return merged


def concat_annotations(annotations_dict: Dict[str, pd.DataFrame]):
    return reduce(user_df_merge, annotations_dict.values())


def generate_data(sample_distributor: SampleDistributor,
                  annotator_dict: dict, num_classes: int) -> pd.DataFrame:
    """Generate a set of anntotations to be tested in annotator
       reliability assessment. Allows control over how good each
       annotator should be, allowing the assessment of the annotation
       framework.

    Args:
        sample_distributor (SampleDistributor): sample distributor giving
            the number of annotations in each project.
        annotator_dict (dict): dictionary of annotator and their
            percentage correctness.
        num_classes (int): the number of possible labels.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    dataset = pd.DataFrame()
    num_annotators = len(annotator_dict)
    annotators = list(annotator_dict.keys())
    for i, current_annotator in enumerate(annotators):
        link_1_annotator = annotators[(i + 1) % num_annotators]
        link_2_annotator = annotators[(i + 2) % num_annotators]

        # current annotator single annotations
        true_labels = np.random.randint(
            0, num_classes, size=sample_distributor.single_annotation_project
        )
        data = pd.DataFrame({"true_label": true_labels})

        data[f"{current_annotator}_label"] = data["true_label"].apply(
            generate_annotator_label,
            correctness_probability=annotator_dict[current_annotator],
            num_classes=num_classes,
        )

        # to satisfy current soft label generator
        data[f"{current_annotator}_confidence"] = 5
        data[f"{current_annotator}_secondary"] = np.nan

        # current annotator reannotaions
        reannotation_sample_indices = data.sample(
            sample_distributor.re_annotation_project
        ).index
        data.loc[
            reannotation_sample_indices, f"re_{current_annotator}_label"
        ] = data.loc[reannotation_sample_indices, "true_label"].apply(
            generate_annotator_label,
            correctness_probability=annotator_dict[current_annotator],
            num_classes=num_classes,
        )
        # to satisfy current soft label generator
        data[f"re_{current_annotator}_confidence"] = 5
        data[f"re_{current_annotator}_secondary"] = np.nan

        # add single annotations to dataset
        dataset = pd.concat([dataset, data], ignore_index=True)

        # current and link_1 annotators
        true_labels = np.random.randint(
            0, num_classes, size=sample_distributor.double_annotation_project
        )
        data = pd.DataFrame({"true_label": true_labels})

        data[f"{current_annotator}_label"] = data["true_label"].apply(
            generate_annotator_label,
            correctness_probability=annotator_dict[current_annotator],
            num_classes=num_classes,
        )
        data[f"{link_1_annotator}_label"] = data["true_label"].apply(
            generate_annotator_label,
            correctness_probability=annotator_dict[link_1_annotator],
            num_classes=num_classes,
        )

        # to satisfy current soft label generator
        data[f"{current_annotator}_confidence"] = 5
        data[f"{current_annotator}_secondary"] = np.nan
        data[f"{link_1_annotator}_confidence"] = 5
        data[f"{link_1_annotator}_secondary"] = np.nan

        # add to dataset
        dataset = pd.concat([dataset, data], ignore_index=True)

        # current and link_2 annotators
        true_labels = np.random.randint(
            0, num_classes, size=sample_distributor.double_annotation_project
        )
        data = pd.DataFrame({"true_label": true_labels})

        data[f"{current_annotator}_label"] = data["true_label"].apply(
            generate_annotator_label,
            correctness_probability=annotator_dict[current_annotator],
            num_classes=num_classes,
        )
        data[f"{link_2_annotator}_label"] = data["true_label"].apply(
            generate_annotator_label,
            correctness_probability=annotator_dict[link_2_annotator],
            num_classes=num_classes,
        )

        # to satisfy current soft label generator
        data[f"{current_annotator}_confidence"] = 5
        data[f"{current_annotator}_secondary"] = np.nan
        data[f"{link_2_annotator}_confidence"] = 5
        data[f"{link_2_annotator}_secondary"] = np.nan

        # add to dataset
        dataset = pd.concat([dataset, data], ignore_index=True)

    return dataset
