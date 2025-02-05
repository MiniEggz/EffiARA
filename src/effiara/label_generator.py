from abc import ABC, abstractmethod

import pandas as pd


class LabelGenerator(ABC):
    """Abstract class for generation of labels for set of annotations.

    This class should be subclassed for each individual annotation
    project. The subclass should override the following methods:
      * add_annotation_prob_labels
      * add_sample_prob_labels
      * add_sample_hard_labels

    That is, create a new file with the following:

    .. code-block:: python

        from effiara import LabelGenerator

        class MyLabelGenerator(LabelGenerator):

            def add_annotation_prob_labels(self, df):
                ...

            def add_sample_prob_labels(self, df, reliability_dict):
                ...

            def add_sample_hard_labels(self, df):
                ...
    """

    def __init__(self,
                 annotators: list,
                 label_mapping: dict):
        self.annotators = annotators
        self.num_annotators = len(self.annotators)
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
