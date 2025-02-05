import os
import argparse

from effiara.annotator_reliability import Annotations
from effiara.data_generator import (
    annotate_samples,
    concat_annotations,
    generate_samples,
)
from effiara.effi_label_generator import EffiLabelGenerator
from effiara.preparation import SampleDistributor, SampleRedistributor

# example for creating set of samples, annotations, and sticking them together
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--usernames",action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs("./data", exist_ok=True)

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
    effiannos.display_agreement_heatmap()

    print("Re-Annotating")
    os.makedirs("./data/redistributed")
    sample_redistributor = SampleRedistributor.from_sample_distributor(sample_distributor)  # noqa
    sample_redistributor.set_project_distribution()
    sample_redistributor.distribute_samples(
        annotations, "./data/redistributed")
    annotate_samples(annotator_dict, "./data/redistributed", num_classes)
    reannotations = concat_annotations("./data/redistributed/annotations",
                                       sample_redistributor.annotators)
    print(reannotations)
    effi_reannos = Annotations(reannotations, label_generator)
    print(effi_reannos.get_reliability_dict())
    effi_reannos.display_agreement_heatmap()
