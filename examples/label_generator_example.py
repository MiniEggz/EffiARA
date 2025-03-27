import numpy as np
import pandas as pd

from effiara.label_generators import LabelGenerator
from effiara.annotator_reliability import Annotations


annotations = pd.DataFrame(
   {"Larry_label":      ["yes", "no", "no", "yes", "yes"],
    "Larry_confidence": [4, 2, 5, 5, 3],
    "Curly_label":      ["no", "no", "yes", "no", "yes"],
    "Curly_confidence": [5, 5, 4, 2, 3],
    "Moe_label":        ["yes", "yes", "yes", "no", "yes"],
    "Moe_confidence":   [2, 5, 3, 4, 3]}
 )
print(annotations)


class ConfidenceLabelGenerator(LabelGenerator):

    def _to_onehot(self, lab):
        # self.label_mapping is an argument to __init__
        onehot = np.zeros(len(self.label_mapping))
        idx = self.label_mapping[lab]
        onehot[idx] = 1.
        return onehot

    def add_annotation_prob_labels(self, df):
        """
        Convert the annotation of a single annotator on a single
        example into a probability distribution over classes.
        """
        conf_scale = np.arange(1, 6)
        # weights are evenly spaced between [0.5 - 0.0]
        weights = np.linspace(0.5, 1.0, 5)
        conf2weight = dict(zip(conf_scale, weights))

        def add_prob_label_to_row(row):
            # self.annotators is an argument to __init__
            for annotator in self.annotators:
                y = self._to_onehot(row[f"{annotator}_label"])
                conf = row[f"{annotator}_confidence"]
                weight = conf2weight[conf]
                row[f"{annotator}_prob"] = np.abs(y - weight)
            return row

        return df.apply(add_prob_label_to_row, axis=1)

    def add_sample_prob_labels(self, df, reliability_dict=None):
        """
        Aggregate annotations from multiple annotators into a single
        probability distribution.
        """
        prob_label_cols = [c for c in df.columns if c.endswith("_prob")]
        if len(prob_label_cols) == 0:
            df = self.add_annotation_prob_labels(df)

        def compute_avg_prob_label(row):
            row["consensus_prob"] = np.mean(row[prob_label_cols])
            return row

        return df.apply(compute_avg_prob_label, axis=1)

    def add_sample_hard_labels(self, df):
        """
        Aggregate annotations from multiple annotators into a single
        'hard' onehot encoding.
        """
        if "consensus_prob" not in df.columns:
            df = self.add_sample_prob_labels(df)

        dfcp = df.copy()
        onehots = np.zeros((len(df), len(self.label_mapping)))
        idxs = df["consensus_prob"].apply(np.argmax)
        onehots[np.arange(len(df)), idxs] = 1
        dfcp["consensus_hard"] = list(onehots)
        return dfcp


label_mapping = {"no": 0, "yes": 1}
annotators = ["Larry", "Curly", "Moe"]
label_generator = ConfidenceLabelGenerator(annotators, label_mapping)
annos = Annotations(annotations,  # we defined this above.
                    label_generator=label_generator,
                    agreement_metric="cosine",
                    agreement_suffix="_prob")
print(annos.df)
annos.display_annotator_graph()
