import re
from abc import ABC, abstractmethod

import numpy as np
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
    @classmethod
    def from_annotations(cls, df: pd.DataFrame, num_classes=None):
        """Initialize from an annotations dataframe

        Args:
            df (pd.DataFrame): annotations, must contain *_label columns.
            num_classes (int): if not None, infer from df.
        """
        # Check for columns with hard labels.
        label_cols = [c for c in df.columns
                      if c != "true_label"
                      and c.endswith("_label")
                      and not c.startswith("re_")]
        if len(label_cols) == 0:
            raise ValueError("No *_label columns found in annotations!")

        # Find annotators from label columns
        user_re = re.compile(r"(re_)?([\w -_]+)_.+")
        annotators = []
        for lc in label_cols:
            user_re_match = user_re.match(lc)
            if user_re_match is None:
                raise ValueError(f"Improperly formatted label column '{lc}'")
            # Using an `if` rather than set() keeps the
            # annotators in the same order as the columns.
            username = user_re_match.group(2)
            if username not in annotators:
                annotators.append(username)

        # Create a default label mapping
        labels = df[label_cols].values.flatten()
        labels_set = set(labels[~np.isnan(labels)].tolist())

        num_classes = num_classes or len(labels_set)
        if num_classes < len(labels_set):
            raise ValueError(f"num_classes ({num_classes}) less than the number of classes found in annotations ({len(labels_set)})!")  # noqa
        # We can't guarantee that all the labels
        # are given in the annotations, so if there's a
        # discrepancy add some placeholder labels.
        if num_classes > len(labels_set):
            num_extra = num_classes - len(labels_set)
            for n in range(num_extra):
                labels_set.add(f"_label{n}")

        label_mapping = dict(zip(labels_set, range(num_classes)))
        return cls(annotators, label_mapping)

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


class DefaultLabelGenerator(LabelGenerator):
    """The most basic LabelGenerator, with support only for hard labels."""

    def add_annotation_prob_labels(self, df):
        return df

    def add_sample_prob_labels(df, reliability_dict):
        return df

    def add_sample_hard_labels(self, df):
        return df
