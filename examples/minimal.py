from effiara.preparation import SampleDistributor
from effiara.annotator_reliability import Annotations
from effiara.label_generator import DefaultLabelGenerator  # noqa

from effiara.data_generator import (generate_samples,
                                    annotate_samples,
                                    concat_annotations)


# Generate some random data to annotate.
num_classes = 3
num_samples = 500
df = generate_samples(num_samples, num_classes, seed=0)

# Name and percentage correctness for each annotator.
annotators = ["Larry", "Curly", "Moe"]
correctness = [0.95, 0.67, 0.58]
annotator_dict = dict(zip(annotators, correctness))
print(annotator_dict)

# Initialize the sample distributor.
# Note that one of the __init__ variables must be None.
sample_distributor = SampleDistributor(
    annotators=annotators,
    num_annotators=len(annotators),
    time_available=10,
    # This is unknown. SampleDistributor will solve for it.
    annotation_rate=None,
    num_samples=num_samples,
    double_proportion=1/3,
    re_proportion=1/2,
)
sample_distributor.set_project_distribution()
print(sample_distributor)
# Distribute the samples to the annotators.
allocations = sample_distributor.distribute_samples(
   df.copy(), all_reannotation=True)

# Generate annotations according to allocations and annotator correctness.
annotated = annotate_samples(allocations, annotator_dict, num_classes)
annotations = concat_annotations(annotated)
print(annotations)

# Compute reliability metrics.
effiannos = Annotations(annotations, num_classes)
# You can also define a label_generator manually like so,
# if you need more advanced functionality.
# label_mapping = {0.0: 0, 1.0: 1, 2.0: 2}
# label_generator = DefaultLabelGenerator(annotators, label_mapping)
# effiannos = Annotations(annotations, num_classes,
#                         label_generator=label_generator)
print(effiannos.get_reliability_dict())

# Edges are inter-annotator reliability
# Nodes are intra-annotator reliability
effiannos.display_annotator_graph()
