# EffiARA <br> (Efficient Annotator Reliability Assessment) <br> Annotation Framework

EffiARA is an annotation framework designed to streamline dataset creation and annotation workflows for machine learning and research. By focusing on efficiency, transparency, and performance, EffiARA empowers users to create high-quality datasets with minimal overhead. Key features include:

* 📦 **Efficiently Distribute Samples**
* 🛠️ **Assemble Datasets**
* 🤝 **Assess Annotator Agreement**
* ⚖️ **Annotator-Specific Sample Weighting in Training**

EffiARA aims to give much-needed structure to a range of annotation tasks within research and industry.

In its current state, the annotation framework is suitable for classification tasks, supporting both
multi-class single-label and multi-class multi-label classification. While it has currently only been
applied in Natural Language Processing datasets ([https://github.com/MiniEggz/ruc-misinfo](https://github.com/MiniEggz/ruc-misinfo) and [https://zenodo.org/records/14659362](https://zenodo.org/records/14659362)), it is
applicable in any setting requiring the distribution of samples and/or the aggregation of annotations.

In Cook, et al. 2024 ([https://doi.org/10.48550/arXiv.2410.14515](https://doi.org/10.48550/arXiv.2410.14515))
we introduced the annotation framework and discuss its applications in assessing annotator reliability
and how this information can be leveraged to weight samples. Annotator reliabillity is assessed through
inter- and intra-annotator agreeement.

* Inter-annotator agreement is calculated pairwise, based on a threshold number of annotations shared
between two annotators.
* Intra-annotator agreement is calculated pairwise, using the annotators' initial and re-annotations.
* Further details on agreement calculation can be seen [here](#agreement).

## Table of Contents
* [Installation](#installation)
* [Framework Components](#framework-components)
* [Modules](#modules)
    * [Agreement](#agreement)
    * [Label Generator](#label-generator)
    * [Preparation](#preparation)
    * [Data Generator](#data-generator)
    * [Annotator Reliability](#annotator-reliability)
* [Usage](#usage)
    * [Sample Distribution](#sample-distribution)
* [Examples](#examples)
    * [Full Pipeline](#full-pipeline)
* [Contribution](#contribution)
* [License](#license)


## Installation

### Current Installation (from source)

1. **Clone the Repository**
```bash
git clone git@github.com:MiniEggz/EffiARA.git
cd EffiARA
```

2. **Install the package**: Make sure pip is installed and run:
```bash
pip install .
```

### Future Installation
When we reach a more stable release, we will make this package available on PyPi,
installable using:
```bash
pip install effiara
```

### Virtual Environments
We recommend installing this tool in a virtual environment, using either
[venv](https://docs.python.org/3/library/venv.html) or a version of Anaconda, such as [miniconda](https://docs.anaconda.com/miniconda/install/).


## Framework Components

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

## Contribution

We warmly welcome contributions to improve EffiARA! If you find that this package doesn't fully suit your needs but could with a small change, we encourage you to contribute to the project. Here's how you can help:

- **General Solutions**: Please try to make your contributions as general as possible, so they can benefit a wide range of tasks and users.
- **Specific Solutions**: Even if your solution addresses a very specific need, don’t hesitate to put in a Pull Request (PR). Others may benefit from it too! When submitting, be sure to include a clear description of the purpose and context of your changes.

### Contribution Guidelines
- All contributions are welcome but must pass the CI/CD pipeline.
- Each PR will require at least one approval before being merged.
- Please ensure your code is well-documented and follows the project’s coding style.

### Need Help?
If you’d like to discuss a potential contribution or have any questions, feel free to contact me directly at **oscook1@sheffield.ac.uk**. I'm happy to help!

We would be hugely greatful for any contribution you are able to provide!


## License

This project is licensed under the **MIT License**.

You are free to:
* Use the code for **any purpose**, including commercial use.
* Modify and distribute the code as long as you include the original license.

A copy of the license is included in this repository. You can also view the full license text [here](https://opensource.org/licenses/MIT).

---

```
MIT License

Copyright (c) 2025 Owen Cook, Jake Vasilakes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
