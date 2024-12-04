import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from effiara.label_generator import LabelGenerator
from effiara.utils import csv_to_array


class TopicLabelGenerator(LabelGenerator):

    def __init__(self, num_annotators: int, label_mapping: dict):
        super().__init__(num_annotators, label_mapping)
        self.mlb = MultiLabelBinarizer(classes=list(label_mapping.keys()))
        self.mlb.fit([])  # predefined fit so call with empty list

    def binarize(self, label):
        # print(self.mlb.transform([label])[0])
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
            f"{prefix}_{i}"
            for i in range(1, self.num_annotators + 1)
            for prefix in ["user", "re_user"]
        ]

        for prefix in all_prefixes:
            row[f"{prefix}_label"] = csv_to_array(row[f"{prefix}_label"])
            if isinstance(row[f"{prefix}_label"], list):
                row[f"{prefix}_bin_label"] = self.binarize(row[f"{prefix}_label"])
            else:
                row[f"{prefix}_bin_label"] = np.nan
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
        # only account for non-reannotations
        prefixes = [f"user_{i}" for i in range(1, self.num_annotators+1)]
        # set label
        row["soft_label"] = np.zeros(len(self.label_mapping))
        reliability_sum = 0
        num_sample_annotators = 0

        # go through each annotator
        for prefix in prefixes:
            # check if this annotator annotate the sample
            if isinstance(row[f"{prefix}_bin_label"], (list, np.ndarray)):
                num_sample_annotators += 1
                reliability_sum += reliability_dict[prefix]
                row["soft_label"] += reliability_dict[prefix] * np.array(row[f"{prefix}_bin_label"]) 

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
