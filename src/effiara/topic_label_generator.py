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
        print(self.mlb.transform([label])[0])
        return self.mlb.transform([label])[0]

    def add_annotation_prob_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binarised labels with a 1 representing a topic's presence."""
        print("doing this")
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
        return df

    def add_sample_hard_labels(self, df) -> pd.DataFrame:
        return df
