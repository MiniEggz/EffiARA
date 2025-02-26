"""Functions for computing agreement metrics."""

import krippendorff
import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from effiara.utils import headings_contain_prob_labels, retrieve_pair_annotations


def pairwise_nominal_krippendorff_agreement(
    pair_df, heading_1, heading_2, label_mapping
):
    """Get the nominal krippendorff agreement between two annotators,
       given two headings for each annotator column containing their
       primary label for each sample.

       Does not require any specific formatting of labels within the columns
       heading_1 and heading_2.

    Args:
        pair_df (pd.DataFrame): dataframe filtered to contain only the samples
                                that allow agreement calculations.
        heading_1 (str): heading of the first column required
                         to calculate agreement.
        heading_2 (str): heading of the second column required
                         to calculate agreement.
        label_mapping (dict): mapping of labels to numeric values.

    Returns:
        float: Krippendorff's Alpha.
    """
    if pair_df[heading_1].isna().any() or pair_df[heading_2].isna().any():
        raise ValueError(
            "One or both of the columns given contain NaN values; the column names may be incorrect or there is an issue with the data."  # noqa
        )

    # convert string categories to numeric values
    pair_df[heading_1 + "_numeric"] = pair_df[heading_1].map(label_mapping)
    pair_df[heading_2 + "_numeric"] = pair_df[heading_2].map(label_mapping)

    if (
        pair_df[heading_1 + "_numeric"].isna().any()
        or pair_df[heading_2 + "_numeric"].isna().any()
    ):
        raise ValueError(
            "Unexpected label found; Please ensure your label mapping is comprehensive."  # noqa
        )

    # turn the pair dataframe into 2d array for Krippendorff calculation
    krippendorff_format_data = (
        pair_df[[heading_1 + "_numeric", heading_2 + "_numeric"]].to_numpy().T
    )
    return krippendorff.alpha(
        reliability_data=krippendorff_format_data, level_of_measurement="nominal"
    )


def pairwise_cohens_kappa_agreement(pair_df, heading_1, heading_2, label_mapping):
    """Cohen's kappa agreement metric between two annotators, given two
       headings for each annotator column containing their primary label
       for each sample.

       Does not require any specific formatting of labels within the columns
       heading_1 and heading_2.

    Args:
        pair_df (pd.DataFrame): dataframe filtered to contain only the samples
                                that allow agreement calculations.
        heading_1 (str): heading of the first column required
                         to calculate agreement.
        heading_2 (str): heading of the second column required
                         to calculate agreement.
        label_mapping (dict): mapping of labels to numeric values.

    Returns:
        float: Cohen's Kappa.
    """
    if pair_df[heading_1].isna().any() or pair_df[heading_2].isna().any():
        raise ValueError(
            "One or both of the columns given contain NaN values; the column names may be incorrect or there is an issue with the data."  # noqa
        )

    if pair_df[heading_1].empty or pair_df[heading_2].empty:
        raise ValueError("One or both of the columns is empty.")

    # convert string categories to numeric values
    pair_df[heading_1 + "_numeric"] = pair_df[heading_1].map(label_mapping)
    pair_df[heading_2 + "_numeric"] = pair_df[heading_2].map(label_mapping)

    if (
        pair_df[heading_1 + "_numeric"].isna().any()
        or pair_df[heading_2 + "_numeric"].isna().any()
    ):
        raise ValueError(
            "Unexpected label found; Please ensure your label mapping is comprehensive."  # noqa
        )

    user_x = pair_df[heading_1 + "_numeric"].to_numpy()
    user_y = pair_df[heading_2 + "_numeric"].to_numpy()

    return cohen_kappa_score(user_x, user_y)


def pairwise_fleiss_kappa_agreement(pair_df, heading_1, heading_2, label_mapping):
    """Fleiss kappa agreement metric between two annotators, given two
       headings for each annotator column containing their primary label
       for each sample.

       Does not require any specific formatting of labels within the columns
       heading_1 and heading_2.

    Args:
        pair_df (pd.DataFrame): dataframe filtered to contain only the samples
                                that allow agreement calculations.
        heading_1 (str): heading of the first column required
                         to calculate agreement.
        heading_2 (str): heading of the second column required
                         to calculate agreement.
        label_mapping (dict): mapping of labels to numeric values.

    Returns:
        float: Fleiss' Kappa.
    """
    if pair_df[heading_1].isna().any() or pair_df[heading_2].isna().any():
        raise ValueError(
            "One or both of the columns given contain NaN values; the column names may be incorrect or there is an issue with the data."  # noqa
        )

    if pair_df[heading_1].empty or pair_df[heading_2].empty:
        raise ValueError("One or both of the columns is empty.")

    # convert string categories to numeric values
    pair_df[heading_1 + "_numeric"] = pair_df[heading_1].map(label_mapping)
    pair_df[heading_2 + "_numeric"] = pair_df[heading_2].map(label_mapping)

    if (
        pair_df[heading_1 + "_numeric"].isna().any()
        or pair_df[heading_2 + "_numeric"].isna().any()
    ):
        raise ValueError(
            "Unexpected label found; Please ensure your label mapping is comprehensive."  # noqa
        )

    fleiss_format_data, _ = aggregate_raters(
        (pair_df[[heading_1 + "_numeric", heading_2 + "_numeric"]].to_numpy())
    )

    return fleiss_kappa(fleiss_format_data, method="fleiss")


def cosine_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two vectors.

    Args:
        vector_a (np.ndarray)
        vector_b (np.ndarray)

    Returns:
        float: cosine similarity between the two vectors.

    Raises:
        ZeroDivisionError: when vector_a or vector_b is the zero vector.
    """
    if not isinstance(vector_a, np.ndarray) or not isinstance(vector_b, np.ndarray):
        raise ValueError("Vectors a and b MUST be of type np.ndarray.")
    if vector_a.shape != vector_b.shape:
        raise ValueError("Vectors a and b must be of the same shape.")

    numerator = np.dot(vector_a, vector_b)
    divisor = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if np.isclose(divisor, 0):
        raise ZeroDivisionError(
            "Zero division has occurred, vector a or b is the zero vector."
        )

    return numerator / divisor


def pairwise_cosine_similarity(pair_df, heading_1, heading_2, num_classes=3):
    """Calculate the cosine similarity between two columns of soft labels.

       Requires the two headings to be formatted as a soft label
       (list or np.array filled with floats summing to 1).

    Args:
        pair_df (pd.DataFrame): data frame containing annotation data.
        heading_1 (str): heading of first column containing soft labels.
        heading_2 (str): heading of second column containing soft labels.

    Returns:
        float: average cosine similarity between the two sets of soft labels.
    """
    if pair_df[heading_1].isna().any() or pair_df[heading_2].isna().any():
        raise ValueError(
            "One or both of the columns given contain NaN values; the column names may be incorrect or there is an issue with the data."  # noqa
        )

    if len(pair_df[heading_1]) == 0 or len(pair_df[heading_2]) == 0:
        raise ValueError("One or both of the columns is empty.")

    if not headings_contain_prob_labels(pair_df, heading_1, heading_2, num_classes):
        raise ValueError("No probabilistic labels found in dataframe.")

    cosine_similarities = pair_df.apply(
        lambda row: cosine_similarity(row[heading_1], row[heading_2]), axis=1
    )
    return np.sum(cosine_similarities) / len(cosine_similarities)


def calculate_krippendorff_alpha_per_label(
    pair_df, annotator_1_col, annotator_2_col, agreement_type="nominal"
):
    """Calculate Krippendorff's alpha for each label and return the average.

       Requires the data in the given columns to be a binarised array of
       each label (i.e. whether the label is present in the given sample).

    Args:
        annotator_1_col (str): column containing the binarised annotations
            for the first annotator.
        annotator_2_col (str): column containing the binarised annotations
            for the second annotator.
        agreement_type (str): type of agreement:
            - nominal
            - ordinal
            - interval
            - ratio

    Returns:
        float: average Krippendorff's alpha across labels.
    """
    # TODO: check that the column is in the correct format
    annotations_1 = np.vstack(pair_df[annotator_1_col].to_numpy())
    annotations_2 = np.vstack(pair_df[annotator_2_col].to_numpy())

    is_annotations_1_numeric = np.issubdtype(annotations_1.dtype, np.number)
    is_annotations_2_numeric = np.issubdtype(annotations_2.dtype, np.number)
    if not is_annotations_1_numeric or not is_annotations_2_numeric:
        raise ValueError("Annotations are not numeric!")

    if annotations_1.shape != annotations_2.shape:
        raise ValueError("Annotation matrices must have the same shape.")

    alpha_values = []
    # calculate alpha for each label
    for i in range(annotations_1.shape[1]):
        # stack the annotations of both annotators
        label_annotations = np.vstack([annotations_1.T[i], annotations_2.T[i]])

        if len(np.unique(label_annotations)) > 1:
            alpha = krippendorff.alpha(
                reliability_data=label_annotations, level_of_measurement=agreement_type
            )
            alpha_values.append(alpha)
        else:
            alpha_values.append(np.nan)

    # calculate average alpha across labels
    average_alpha = np.nanmean(alpha_values)

    return average_alpha


def pairwise_agreement(
    df,
    user_x,
    user_y,
    label_mapping,
    num_classes,
    metric="krippendorff",
    agreement_type="nominal",
    label_suffix="_label",
):
    """Get the pairwise annotator agreement given the full dataframe.

    Args:
        df (pd.DataFrame): full dataframe containing the whole dataset.
        user_x (str): name of the user in the form user_x.
        user_y (str): name of the user in the form user_y.
        metric (str): agreement metric to use for
                      inter-/intra-annotator agreement:
            - krippendorff: nominal krippendorff's alpha similarity metric
                            on hard labels only.
            - cohen: nominal cohen's kappa similarity metric on hard
                     labels only.
            - fleiss: nominal fleiss kappa similarity metric on hard
                      labels only.
            - multi_krippendorff: krippendorff similarity by label for
                                  multilabel classification.
            - cosine: the cosine similarity metric to be used on soft labels.
        agreement_type (str):
            type of agreement:
                - nominal
                - ordinal
                - interval
                - ratio
            NOTE: currently only working for multi_krippendorff.
        label_suffix (str): suffix for the label being compared.

    Returns:
        float: agreement between user_x and user_y.
    """
    pair_df = retrieve_pair_annotations(df, user_x, user_y)
    if metric == "krippendorff":
        return pairwise_nominal_krippendorff_agreement(
            pair_df, user_x + label_suffix, user_y + label_suffix, label_mapping
        )
    elif metric == "cohen":
        return pairwise_cohens_kappa_agreement(
            pair_df, user_x + label_suffix, user_y + label_suffix, label_mapping
        )
    elif metric == "fleiss":
        return pairwise_fleiss_kappa_agreement(
            pair_df, user_x + label_suffix, user_y + label_suffix, label_mapping
        )
    elif metric == "cosine":
        return pairwise_cosine_similarity(
            pair_df,
            user_x + label_suffix,
            user_y + label_suffix,
            num_classes=num_classes,
        )
    elif metric == "multi_krippendorff":
        return calculate_krippendorff_alpha_per_label(
            pair_df,
            user_x + label_suffix,
            user_y + label_suffix,
            agreement_type=agreement_type,
        )
    else:
        raise ValueError(f"The metric {metric} was not recognised.")
