"""Label generator as in https://doi.org/10.48550/arXiv.2410.14515."""

import numpy as np
import pandas as pd

from effiara.label_generator import LabelGenerator


def convert_confidence(confidence, num_classes=3):
    """
    Convert an annotator's confidence value in the range 1-5
    to the range 0-1.

    Args:
        confidence (int): the annotator's confidence in their primary label.

    Returns:
        int: confidence value in the range 0-1
    """
    return (1 / num_classes) + (
        ((num_classes - 1) / num_classes) * ((confidence - 1) / 4)
    )


class EffiLabelGenerator(LabelGenerator):

    def add_annotation_prob_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add soft labels to the dataframe and return the new dataframe.

        Args:
            df (pd.DataFrame): input dataframe.

        Returns:
            pd.DataFrame: dataframe containing generated soft labels for each user.
        """
        return pd.DataFrame(
            df.apply(
                lambda row: self._add_row_soft_labels(
                    row,
                ),
                axis=1,
            )
        )

    def add_sample_prob_labels(self, df, reliability_dict):
        """Generate soft labels for each row in an annotated DataFrame.
           Also generates the sample weightings based on annotator reliability scores.

        Args:
            df (pd.DataFrame): input dataframe.

        Returns:
            pd.DataFrame: final dataframe with soft labels and sample weightings.
        """
        return pd.DataFrame(
            pd.DataFrame(
                df.apply(
                    lambda row: self._create_row_final_soft_label(
                        row, reliability_dict
                    ),
                    axis=1,
                )
            )
        )

    def add_sample_hard_labels(self, df):
        """Generate hard labels for each row in an annotated DataFrame.

        Args:
            input_df (pd.DataFrame): The annotated DataFrame

        Returns:
            pd.DataFrame: final DataFrame with hard labels.
        """
        return df.apply(lambda row: self._create_row_hard_label(row), axis=1)

    def _create_row_hard_label(self, row):
        """Creates the hard label for the row given there is already
           a soft label in place. This should always be the case.

        Args:
            row (pd.Series): row of the dataframe.

        Returns:
            pd.Series: modified row with the extra "hard_label"
                column.

        Raises:
            ValueError: where no soft label is available for this row -
                this should never be the case.
        """
        soft_label = row["soft_label"]
        soft_label = np.array(soft_label)
        hard_label = np.zeros_like(soft_label)
        # first occurence of max num
        hard_label[np.argmax(soft_label)] = 1
        row["hard_label"] = hard_label
        return row

    def _create_row_final_soft_label(
        self,
        row,
        reliability_dict,
    ):
        """Create final soft label for row with either one or two annotations.
           Adds this final soft label to the row "soft_label" and also
           calculates the final sample weight based on the reliability of the
           annotators that have annotated the sample.

        Args:
            row (pd.Series): single row of the annotation dataframe.
            reliability_dict (dict): dict containing username: reliability pairs.

        Returns:
            pd.Series: the updated row containing columns of final_soft_label and sample_weight.
        """
        # TODO: convert to list comprehension?
        row_annotators = []
        # loop through each annotator
        for user in self.annotators:
            # check if user has a soft label column that isn't NaN
            if isinstance(row[f"{user}_soft_label"], np.ndarray):
                # if so, append prefix to row_annotators list
                row_annotators.append(prefix)

        if len(row_annotators) == 0:
            raise ValueError(
                f"Row number {row.name} does not have annotations, remove this row from the dataset."
            )
        elif len(row_annotators) == 1:
            row["soft_label"] = row[f"{row_annotators[0]}_soft_label"]
            row["sample_weight"] = reliability_dict[row_annotators[0]]
        elif len(row_annotators) == 2:
            # weighted average of labels based on annotator reliability
            soft_label_1 = row[f"{row_annotators[0]}_soft_label"]
            reliability_1 = reliability_dict[row_annotators[0]]
            soft_label_2 = row[f"{row_annotators[1]}_soft_label"]
            reliability_2 = reliability_dict[row_annotators[1]]
            reliability_sum = reliability_1 + reliability_2
            row["soft_label"] = (reliability_1 / reliability_sum) * soft_label_1 + (
                reliability_2 / reliability_sum
            ) * soft_label_2
            row["sample_weight"] = reliability_sum / 2
        else:
            raise ValueError("This row has more than 2 annotators.")
        return row

    def _add_row_soft_labels(self, row):
        """Add all possible soft labels to the given row.

        Args:
            row (pd.Series): a single row of the full annotation DataFrame.

        Returns:
            row [pd.Series]: Row containing generated soft labels.
        """
        for user in self.annotators:
            valid_user_prefixes = [user, f"re_{user}"]

            for prefix in valid_user_prefixes:
                row[f"{prefix}_soft_label"] = self._create_user_soft_label(
                    row,
                    prefix,
                )

        return row

    def _create_user_soft_label(self, row, user_prefix):
        """Generate the soft label of the given user.

        Args:
            row (pd.Series): a single row of the full annotation DataFrame.
            user_prefix (str): user's prefix in the form username or re_username.

        Returns:
            np.ndarray: vector of soft labels, with each dimension representing probability
                        of the sample fitting the class.
        """

        soft_label = np.zeros(self.num_classes)

        primary_label = row[f"{user_prefix}_label"]

        # user has not made an annotation
        if pd.isna(primary_label):
            return np.nan

        confidence_col = f"{user_prefix}_confidence"

        # get stances and confidence from row
        secondary_label = row[f"{user_prefix}_secondary"]
        primary_confidence = int(row[confidence_col])

        # retrieve index of label
        primary_index = self.label_mapping[primary_label]

        # update primary label with the given confidence
        soft_label[primary_index] = convert_confidence(primary_confidence)

        if pd.isna(secondary_label) or secondary_label == "NotAva":
            # redistribute uniformly
            for i in range(len(soft_label)):
                if i != primary_index:
                    soft_label[i] = (1 - soft_label[primary_index]) / (
                        self.num_classes - 1
                    )
        elif primary_label == secondary_label:
            # edge case where secondary has been changed
            # to be the same label - is equal to all
            # being redistributed to stance 2
            soft_label[primary_index] = 1
        else:
            # redistribute all remaining to stance 2
            secondary_index = self.label_mapping[secondary_label]
            soft_label[secondary_index] = 1 - soft_label[primary_index]

            # if secondary is higher than primary
            if soft_label[secondary_index] > soft_label[primary_index]:
                soft_label[secondary_index] = soft_label[primary_index]
                # case where secondary is bigger than primary, set secondary equal and redistribute rest to other label
                for i in range(len(soft_label)):
                    remaining_probability = (
                        1 - soft_label[primary_index] - soft_label[secondary_index]
                    )
                    if i != primary_index and i != secondary_index:
                        soft_label[i] = remaining_probability / (self.num_classes - 2)

        return soft_label
