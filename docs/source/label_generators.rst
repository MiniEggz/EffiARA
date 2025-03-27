Creating a LabelGenerator
=========================

Annotation tasks are often more complex than simply making each annotator assign a single
label to each data point. When this is the case, it is necessary to create a custom
class that inherits from :code:`LabelGenerator` to transform the raw annotations into numeric encodings
that can be used to compute agreement.

Let's give an example. As before, we have multiple annotators per example, but we've 
also instructed them to provide a subjective confidence score regarding their annotations.
The confidence score is a Likert scale from 1-5, where 5 means "absolutely certain".

.. code-block:: python

   import numpy as np
   import pandas as pd

   annotations = pd.DataFrame(
      {"Larry_label":      ["yes", "no", "no", "yes", "yes"],
       "Larry_confidence": [4, 2, 5, 5, 3],
       "Curly_label":      ["no", "no", "yes", "no", "yes"],
       "Curly_confidence": [5, 5, 4, 2, 3],
       "Moe_label":        ["yes", "yes", "yes", "no", "yes"],
       "Moe_confidence":   [2, 5, 3, 4, 3]}
    )
   print(annotations)

     Larry_label  Larry_confidence Curly_label  Curly_confidence Moe_label  Moe_confidence
   0         yes                 4          no                 5       yes               2
   1          no                 2          no                 5       yes               5
   2          no                 5         yes                 4       yes               3
   3         yes                 5          no                 2        no               4
   4         yes                 3         yes                 3       yes               3


We have to determine three things:

 1. How to combine a single annotator's label and confidence score into a probability.
 2. How to combine the individual annotator's probabilities into a single probability for a sample. 
 3. Same as 2, but determine a single "hard" label as consensus.

For item 1, let's make it so a confidence of 5 doesn't change the label at all, while a confidence of
1 results in a uniform distribution. The other confidence levels will fall between these.

For item 2, we'll average the probabilites from item 1 across annotators. For item 3, we'll just take
the argmax of 2.

These are implemented below as the :code:`add_annotation_prob_labels`, :code:`add_sample_prob_labels`, and :code:`add_sample_hard_labels`
methods of our custom :code:`ConfidenceLabelGenerator`.


**confidence_label_generator.py**

.. code-block:: python

   import numpy as np
   import pandas as pd
   from effiara.label_generators import LabelGenerator


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


Once all this is done, we can pass an instance of our :code:`ConfidenceLabelGenerator`
to the :code:`Annotations` class to use the new labels when computing agreement.

.. code-block:: python

   from effiara.annotator_reliability import Annotations
   from .confidence_label_generator import ConfidenceLabelGenerator  # The class we defined above.

   label_mapping = {"no": 0, "yes": 1}
   annotators = ["Larry", "Curly", "Moe"]
   label_generator = ConfidenceLabelGenerator(annotators, label_mapping)
   annos = Annotations(annotations,  # we defined this above.
                       label_generator=label_generator,
                       agreement_metric="cosine",
                       agreement_suffix="_prob")  # Compute agreement from these columns
   print(annos)
