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
