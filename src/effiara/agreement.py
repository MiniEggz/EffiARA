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
        heading_1 (str): heading of the first column required to calculate agreement.
        heading_2 (str): heading of the second column required to calculate agreement.
        label_mapping (dict): mapping of labels to numeric values.

    Returns:
        float: Krippendorff's Alpha.
    """
    if pair_df[heading_1].isna().any() or pair_df[heading_2].isna().any():
        raise ValueError(
            "One or both of the columns given contain NaN values; the column names may be incorrect or there is an issue with the data."
        )

    # convert string categories to numeric values
    pair_df[heading_1 + "_numeric"] = pair_df[heading_1].map(label_mapping)
    pair_df[heading_2 + "_numeric"] = pair_df[heading_2].map(label_mapping)

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
        heading_1 (str): heading of the first column required to calculate agreement.
        heading_2 (str): heading of the second column required to calculate agreement.
        label_mapping (dict): mapping of labels to numeric values.

    Returns:
        float: Cohen's Kappa.
    """
    # convert string categories to numeric values
    pair_df[heading_1 + "_numeric"] = pair_df[heading_1].map(label_mapping)
    pair_df[heading_2 + "_numeric"] = pair_df[heading_2].map(label_mapping)

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
        heading_1 (str): heading of the first column required to calculate agreement.
        heading_2 (str): heading of the second column required to calculate agreement.
        label_mapping (dict): mapping of labels to numeric values.

    Returns:
        float: Fleiss' Kappa.
    """
    # convert string categories to numeric values
    pair_df[heading_1 + "_numeric"] = pair_df[heading_1].map(label_mapping)
    pair_df[heading_2 + "_numeric"] = pair_df[heading_2].map(label_mapping)

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
    """
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def pairwise_cosine_similarity(pair_df, heading_1, heading_2, num_classes=3):
    """Calculate the pairwise cosine similarity between two columns of soft labels.

       Requires the two headings to be formatted as a soft label (list / np.array
       filled with floats summing to 1).

    Args:
        pair_df (pd.DataFrame): data frame containing annotation data.
        heading_1 (str): heading of first column containing soft labels.
        heading_2 (str): heading of second column containing soft labels.

    Returns:
        float: average cosine similarity between the two sets of soft labels.
    """
    if pair_df[heading_1].isna().any() or pair_df[heading_2].isna().any():
        raise ValueError(
            "One or both of the columns given contain NaN values; the column names may be incorrect or there is an issue with the data."
        )

    if not headings_contain_prob_labels(pair_df, heading_1, heading_2, num_classes):
        raise Exception(
            "There has been an issue in generating the soft labels in the dataset."
        )

    cosine_similarities = pair_df.apply(
        lambda row: cosine_similarity(row[heading_1], row[heading_2]), axis=1
    )
    return np.sum(cosine_similarities) / len(cosine_similarities)


def calculate_krippendorff_alpha_per_label(pair_df, annotator_1_col, annotator_2_col):
    """Calculate Krippendorff's alpha for each label and return the average.

       Requires the data in the given columns to be a binarised array of
       each label (i.e. whether the label is present in the given sample).

    Args:
        annotator_1_col (str): column containing the binarised annotations
            for the first annotator.
        annotator_2_col (str): column containing the binarised annotations
            for the second annotator.

    Returns:
        float: average Krippendorff's alpha across labels.
    """
    binarized_annotations_1 = np.vstack(pair_df[annotator_1_col].to_numpy())
    binarized_annotations_2 = np.vstack(pair_df[annotator_2_col].to_numpy())
    assert (
        binarized_annotations_1.shape == binarized_annotations_2.shape
    ), "Annotation matrices must have the same shape"

    alpha_values = []
    # calculate alpha for each label
    for i in range(binarized_annotations_1.shape[1]):
        # stack the annotations of both annotators
        label_annotations = np.vstack(
            [binarized_annotations_1.T[i], binarized_annotations_2.T[i]]
        )

        if len(np.unique(label_annotations)) > 1:
            alpha = krippendorff.alpha(
                reliability_data=label_annotations, level_of_measurement="nominal"
            )
            alpha_values.append(alpha)
        else:
            alpha_values.append(np.nan)

    # calculate average alpha across labels
    average_alpha = np.nanmean(alpha_values)

    return average_alpha


def pairwise_agreement(
    df, user_x, user_y, label_mapping, num_classes, metric="krippendorff"
):
    """Get the pairwise annotator agreement given the full dataframe.

    Args:
        df (pd.DataFrame): full dataframe containing the whole dataset.
        user_x (str): name of the user in the form user_x.
        user_y (str): name of the user in the form user_y.
        metric (str): agreement metric to use for inter-/intra-annotator agreement:
            - krippendorff: nominal krippendorff's alpha similarity metric on hard labels only.
            - cohen: nominal cohen's kappa similarity metric on hard labels only.
            - fleiss: nominal fleiss kappa similarity metric on hard labels only.
            - multi_krippendorff: krippendorff similarity by label for multilabel classification.
            - cosine: the cosine similarity metric to be used on soft labels.

    Returns:
        float: agreement between user_x and user_y.
    """
    pair_df = retrieve_pair_annotations(df, user_x, user_y)
    if metric == "krippendorff":
        return pairwise_nominal_krippendorff_agreement(
            pair_df, user_x + "_label", user_y + "_label", label_mapping
        )
    elif metric == "cohen":
        return pairwise_cohens_kappa_agreement(
            pair_df, user_x + "_label", user_y + "_label", label_mapping
        )
    elif metric == "fleiss":
        return pairwise_fleiss_kappa_agreement(
            pair_df, user_x + "_label", user_y + "_label", label_mapping
        )
    elif metric == "cosine":
        return pairwise_cosine_similarity(
            pair_df,
            user_x + "_soft_label",
            user_y + "_soft_label",
            num_classes=num_classes,
        )
    elif metric == "multi_krippendorff":
        return calculate_krippendorff_alpha_per_label(
            pair_df,
            user_x + "_bin_label",
            user_y + "_bin_label",
        )
    else:
        raise ValueError(f"The metric {metric} was not recognised.")
