from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from effiara.label_generators import LabelGenerator
from effiara.utils import csv_to_array


class TopicLabelGenerator(LabelGenerator):

    def __init__(self, annotators: List[str], label_mapping: dict):
        super().__init__(annotators, label_mapping)
        self.mlb = MultiLabelBinarizer(classes=list(label_mapping.keys()))
        self.mlb.fit([])  # predefined fit so call with empty list

    def binarize(self, label):
        return self.mlb.transform([label])[0]

    def add_annotation_prob_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binarised labels with a 1 representing a topic's presence."""
        return pd.DataFrame(
            df.apply(
                lambda row: self._add_row_annotation_bin_labels(
                    row,
                ),
                axis=1,
            )
        )

    def _add_row_annotation_bin_labels(self, row):
        all_prefixes = [
            f"{anno_prefix}{annotator}"
            for anno_prefix in ["", "re_"]
            for annotator in self.annotators
        ]

        for prefix in all_prefixes:
            row[f"{prefix}_label"] = csv_to_array(row[f"{prefix}_label"])
            if isinstance(row[f"{prefix}_label"], list):
                row[f"{prefix}_bin_label"] = self.binarize(row[f"{prefix}_label"])
            else:
                row[f"{prefix}_bin_label"] = np.nan

        if "gold" in row:
            row["gold"] = csv_to_array(row["gold"])
            row["gold"] = self.binarize(row["gold"])

        return row

    def add_sample_prob_labels(
        self, df: pd.DataFrame, reliability_dict: dict
    ) -> pd.DataFrame:
        return pd.DataFrame(
            df.apply(
                lambda row: self._add_row_sample_prob_labels(
                    row,
                    reliability_dict,
                ),
                axis=1,
            )
        )

    def _add_row_sample_prob_labels(self, row: pd.Series, reliability_dict: dict):
        # TODO: maybe add in something to account for reannotations
        # set label
        row["soft_label"] = np.zeros(len(self.label_mapping))
        reliability_sum = 0
        num_sample_annotators = 0

        # go through each annotator (only non-reannotations)
        for prefix in self.annotators:
            # check if this annotator annotate the sample
            if isinstance(row[f"{prefix}_bin_label"], (list, np.ndarray)):
                num_sample_annotators += 1
                reliability_sum += reliability_dict[prefix]
                bin_label = np.array(row[f"{prefix}_bin_label"])
                row["soft_label"] += reliability_dict[prefix] * bin_label

        # clip soft label (min=0, max=1)
        row["soft_label"] = np.clip(row["soft_label"] / reliability_sum, 0, 1)
        row["sample_weight"] = reliability_sum / num_sample_annotators

        return row

    def add_sample_hard_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            df.apply(
                lambda row: self._add_row_hard_label(
                    row,
                ),
                axis=1,
            )
        )

    def _add_row_hard_label(self, row: pd.Series) -> pd.Series:
        row["hard_label"] = np.where(row["soft_label"] >= 0.5, 1, 0)
        return row
