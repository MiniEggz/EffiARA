from abc import ABC, abstractmethod

import pandas as pd


class LabelGenerator(ABC):
    """Abstract class for generation of labels for set of annotations."""

    def __init__(self,
                 num_annotators: int,
                 label_mapping: dict,
                 annotators: list = None):
        self.num_annotators = num_annotators
        # Set annotators to list of ints if not given.
        if annotators is None:
            self.annotators = list(range(1, self.num_annotators + 1))
        else:
            self.annotators = annotators
            if len(self.annotators) != self.num_annotators:
                raise ValueError(f"Number of annotator names ({len(annotators)}) != num_annotators ({num_annotators})!")
        self.label_mapping = label_mapping
        self.num_classes = len(label_mapping)

    @abstractmethod
    def add_annotation_prob_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add probability distribution (soft) labels
           to each individual annotation.

        Args:
            df (pd.DataFrame): dataframe with all annotation
                data to add probability label column to.

        Returns:
            (pd.DataFrame): dataframe with added labels.
        """
        pass

    @abstractmethod
    def add_sample_prob_labels(
        self, df: pd.DataFrame, reliability_dict: dict
    ) -> pd.DataFrame:
        """Add probability distribution (soft) labels
           to each individual sample, likely using some
           combination of annotation probability
           labels. Can optionally add a sample_weight
           column to weight samples in training based
           on annotator reliability.

        Args:
            df (pd.DataFrame): dataframe with all annotation
                data to add probability label column to.
            reliability_dict (dict): dict of each annotator and
                their reliability score.

        Returns:
            (pd.DataFrame): dataframe with added labels.
        """
        pass

    @abstractmethod
    def add_sample_hard_labels(self, df) -> pd.DataFrame:
        """Implemented to give each sample a one-hot
           hard label for use in the classification
           task.

        Args:
            df (pd.DataFrame): dataframe with all annotation
                data to add probability label column to.

        Returns:
            (pd.DataFrame): dataframe with added labels.
        """
        pass
