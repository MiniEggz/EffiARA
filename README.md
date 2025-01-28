# EffiARA
Python implementation of the Efficient Annotator Reliability Assessment (EffiARA) annotation framework for distributing samples and assessing annotator reliability using inter- and intra-annotator agreement.

Link to paper introducing EffiARA: [https://doi.org/10.48550/arXiv.2410.14515](https://doi.org/10.48550/arXiv.2410.14515).


## Components of the EffiARA annotation framework

The EffiARA annotation framework handles:
* Sample distribution between a number of annotators
* Inter-annotator agreement calculation
* Intra-annotator agreement calculation
* Overall annotator reliability calculation based on weighting of inter- and intra-annotator agreement
* Graphical representation of annotator agreement and reliability

## Modules

### Agreement

The agreement module contains a number of pairwise agreement metrics, including Krippendorff’s Alpha, Cohen’s Kappa, Fleiss’
Kappa and the multi-label Krippendorff’s Alpha. These pairwise agreement metrics can be used through the `pairwise_agreement`
function, which takes the full dataframe (dataset), the two users to obtain agreement between, the label mapping used, and the metric.
The metric is entered as one of the strings below.

```
* krippendorff:
  nominal krippendorff's alpha similarity metric on
  hard labels only.
* cohen:
  nominal cohen's kappa similarity metric on
  hard labels only
* fleiss:
  nominal fleiss kappa similarity metric on hard
  labels only.
* multi_krippendorff:
  krippendorff similarity by label for multilabel
  classification.
* cosine:
  the cosine similarity metric to be used on soft labels.
```

There is currently no way to add your own agreement metric without contributing to the repository.
Please add an issue or create a PR if you would like another agreement metric to be added.


### Label Generator
The label generator allows the EffiARA annotation framework calculations to fit your data. As
some annotations may be structured differently to others, it is possible to create a custom
class extending LabelGenerator.

The only requirements are that it msut have the properties
set in the constructor: `num_annotators` (number of annotators in the dataset), `label_mapping`
(mapping of labels to their numeric counterpart), and `num_classes` (which can be deduced from
the label mapping); it must also have the three methods: `add_annotation_prob_labels` which
adds probability or soft labels to each annotation, `add_sample_prob_labels` which calculates
a final probability or soft label for each sample (aggregating multiple annotations on double-annotated
samples), and `add_sample_hard_labels` which gives each sample a hard label (generally used for creating
test sets).

Two example label generators have been added to the first release, including EffiLabelGenerator
(as used in https://doi.org/10.48550/arXiv.2410.14515) and TopicLabelGenerator for multi-label
topic annotation.


### Preparation

The preparation module contains the `SampleDistributor` class, which can be instantiated
to calculate the number of samples, number of annotators needed, or time required. Once all
annotation parameters (`num_annotators`, `time_available`, `annotation_rate`, `num_samples`,
`double_proportion`, `re_proportion`) have been set, data samples can be distributed to annotators
using the EffiARA annotation framework. To distribute samples, the `distribute_samples` method is
used. Each sample is sampled without replacement, so there should be no repeated samples in
different single- or double-annotated sets.

Once preparation is complete, annotation can take place.


### Data Generator

The data generator module allows you to create some quick synthetic data to test the
EffiARA annotation framework with. You can generate a set of any number of samples using the
`generate_samples` method, passing a sample distributor and the number of classes.
`annotate_samples` can then be used to simulate the annotation process, passing in the sample
distributor, the annotator dictionary containing the average accuracy of each annotator,
the directory path to the generated samples, and the number of classes.

Once annotation is complete, `concat_annotations` can be used to create one whole dataset.

This process is meant to simulate the process of having data samples distributed and saved to
individual annotator CSV files, each annotator making their annotations, and putting them back
into one complete dataset. Once the data generator has completed its annotations and made one
dataset, it can be used to test the annotator reliability calculations.


### Annotator Reliability

The annotator reliability module handles a large amount of the annotation processing.
The `Annotations` class requires the dataset (as a DataFrame), the label generator,
agreement metric (to be used in the `pairwise_agreement` function), and `merge_labels` which
defaults to `None` but can be set to a dictionary with keys for the labels to keep and values
the label to replace with the key.

This class handles replacing labels, generating final labels and sample weights, using the
label generator passed in to calculate the final labels, creating the annotator graph (from
which annotator reliability can be calculated), calculating inter- and intra-annotator
agreement, and finally combining those metrics to calculate the individual annotator
reliability metrics.

## Usage

### Sample Distribution
To set up sample distribution, use a `SampleDistributor` object. This will handle the
sample distribution calculation, given all but one of the following variables
* num_annotators (n)
* time_available (t)
* annotation_rate (rho)
* num_samples (k)
* double_proportion (d)
* re_proportion (r)

Declare the `SampleDistributor` object:

```python
sample_distributor = SampleDistributor()
```

Once that is declared, it is possible to complete the sample distribution variables
using the equation from [https://doi.org/10.48550/arXiv.2410.14515](https://doi.org/10.48550/arXiv.2410.14515).
Generally, this equation will be used to understand the number of samples, time available, or number
of annotators needed.
```python
sample_distributor.get_variables(
    num_annotators=6,
    time_available=10,
    annotation_rate=60,
    #num_samples=2160,
    double_proportion=1/3,
    re_proportion=1/2,
)
```

Once this is complete, calculate the project distributions using the variables:
```python
sample_distributor.set_project_distribution()
```

Now the distribution has been calculated, it is possible to distribute the samples.
For an idea of how this works, you can declare an example dataframe, or use your
data. You may want to copy the dataframe if you want to preserve a full version
of it.
```python
df = sample_distributor.create_example_distribution_df()
sample_distributor.distribute_samples(df.copy(), "/path/to/dir")
```

## Examples

### Full Pipeline
To see an example of the full pipeline, see `examples/data_generation_split.py`. This example
contains usage of the `SampleDistributor`, the `data_generator` module, the `EffiLabelGenerator`,
and the `Annotations` class to calculate annotator reliability.

```python
import os

from effiara.annotator_reliability import Annotations
from effiara.data_generator import (
    annotate_samples,
    concat_annotations,
    generate_samples,
)
from effiara.effi_label_generator import EffiLabelGenerator
from effiara.preparation import SampleDistributor

# example for creating set of samples, annotations, and sticking them together
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Percentage correctness for each annotator.
    annotators = None
    # If annotators is None, names are set to integers in SampleDistributor.
    #annotators = ["aa", "bb", "cc", "dd", "ee", "ff"]
    correctness = [0.95, 0.67, 0.58, 0.63, 0.995, 0.45]

    sample_distributor = SampleDistributor(
        annotators=annotators,
        num_annotators=len(correctness),
        time_available=10,
        annotation_rate=60,
        # num_samples=2160,
        double_proportion=1 / 3,
        re_proportion=1 / 2,
    )
    sample_distributor.set_project_distribution()
    print(sample_distributor)

    num_classes = 3
    df = generate_samples(sample_distributor, num_classes, seed=0)
    sample_distributor.distribute_samples(df.copy(), "./data", all_reannotation=True)

    annotator_dict = dict(zip(sample_distributor.annotators, correctness))
    print(annotator_dict)
    annotate_samples(annotator_dict, "./data", num_classes)
    annotations = concat_annotations("./data/annotations", sample_distributor.annotators)
    print(annotations)

    label_mapping = {0.0: 0, 1.0: 1, 2.0: 2}
    label_generator = EffiLabelGenerator(sample_distributor.annotators, label_mapping)
    effiannos = Annotations(annotations, label_generator)
    print(effiannos.get_reliability_dict())
    effiannos.display_annotator_graph()
    # Equivalent to the graph, but as a heatmap
    effiannos.display_agreement_heatmap()
    # Agreements between two subsets of annotators
    effiannos.display_agreement_heatmap(
            annotators=effiannos.annotators[:4],
            other_annotators=effiannos.annotators[3:])
```
