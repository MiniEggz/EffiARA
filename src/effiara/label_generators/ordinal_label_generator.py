import pandas as pd
import numpy as np

from effiara.label_generators import LabelGenerator


class OrdinalLabelGenerator(LabelGenerator):

    def __init__(self, annotators: list, label_mapping: dict, label_suffixes=["_label"]):
        super().__init__(annotators, label_mapping)
        self.label_suffixes = label_suffixes

    def add_annotation_prob_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add annotations as an np array [lab1, lab2, ...].

        Args:
            df (pd.DataFrame): dataframe with all annotation
                data to add probability label column to.

        Returns:
            (pd.DataFrame): dataframe with added labels.
        """
        return pd.DataFrame(df.apply(self._add_row_soft_label, axis=1))

    def _add_row_soft_label(self, row) -> pd.Series:
        for annotator in self.annotators:

            cols_to_check = [f"{annotator}{suffix}" for suffix in self.label_suffixes]

            # check cols are all there
            if any(col not in row.index for col in cols_to_check):
                raise ValueError(f"At least one suffix is not in the dataframe for all annotators.")
            
            # add label to non-null
            if row[cols_to_check].notna().all():
                values_list = [self.label_mapping[val] for val in list(row[cols_to_check])]
                row[f"{annotator}_label"] = np.array(values_list, dtype=int)

        return row
    
    def add_sample_prob_labels(self, df: pd.DataFrame, reliability_dict: dict) -> pd.DataFrame:
        return df

    def add_sample_hard_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    


