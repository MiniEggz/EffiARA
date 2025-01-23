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

    annotator_dict = {
        "user_aa": 0.95,
        "user_bb": 0.67,
        "user_cc": 0.58,
        "user_dd": 0.63,
        "user_ee": 0.995,
        "user_ff": 0.45,
    }
    annotator_names = [k.split('_')[-1] for k in annotator_dict.keys()]

    sample_distributor = SampleDistributor(
        #num_annotators=len(annotator_dict),
        annotators=annotator_names,
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

    annotate_samples(annotator_dict, "./data", num_classes)
    annotations = concat_annotations("./data/annotations", annotator_names)
    print(annotations)

    label_mapping = {0.0: 0, 1.0: 1, 2.0: 2}
    label_generator = EffiLabelGenerator(6, label_mapping, annotators=annotator_names)
    effiannos = Annotations(annotations, label_generator)
    print(effiannos.get_reliability_dict())
    #effiannos.display_annotator_graph()
    # Equivalent to the graph, but as a heatmap
    effiannos.display_agreement_heatmap()
    # Agreements between two subsets of annotators
    effiannos.display_agreement_heatmap(
            annotators=effiannos.annotators[:4],
            other_annotators=effiannos.annotators[3:])
