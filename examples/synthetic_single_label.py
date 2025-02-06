import argparse

from effiara.annotator_reliability import Annotations
from effiara.data_generator import (
    annotate_samples,
    concat_annotations,
    generate_samples,
)
from effiara.label_generators.effi_label_generator import EffiLabelGenerator
from effiara.preparation import SampleDistributor


parser = argparse.ArgumentParser()
parser.add_argument("--usernames", action="store_true", default=False)
args = parser.parse_args()

num_classes = 3
num_samples = 2160
df = generate_samples(num_samples, num_classes, seed=0)

# If annotators is None, names are set to integers in SampleDistributor.
annotators = None
if args.usernames is True:
    annotators = ["aa", "bb", "cc", "dd", "ee", "ff"]
# Percentage correctness for each annotator.
correctness = [0.95, 0.67, 0.58, 0.63, 0.995, 0.45]

sample_distributor = SampleDistributor(
    annotators=annotators,
    num_annotators=len(correctness),
    time_available=10,
    annotation_rate=None,
    num_samples=num_samples,
    double_proportion=1/3,
    re_proportion=1/2,
)
sample_distributor.set_project_distribution()
print(sample_distributor)
allocations = sample_distributor.distribute_samples(
    df.copy(), all_reannotation=True)

annotator_dict = dict(zip(sample_distributor.annotators, correctness))
annotated = annotate_samples(allocations, annotator_dict, num_classes)
annotations = concat_annotations(annotated)
print(annotations)

label_mapping = {0.0: 0, 1.0: 1, 2.0: 2}
label_generator = EffiLabelGenerator(
    sample_distributor.annotators, label_mapping)
effiannos = Annotations(annotations, len(label_mapping), label_generator)
print(effiannos.get_reliability_dict())
effiannos.display_annotator_graph()
# Equivalent to the graph, but as a heatmap
effiannos.display_agreement_heatmap()
# Agreements between two subsets of annotators
effiannos.display_agreement_heatmap(
        annotators=effiannos.annotators[:4],
        other_annotators=effiannos.annotators[3:])
