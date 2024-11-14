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

    annotator_dict = {
        "user_1": 0.95,
        "user_2": 0.67,
        "user_3": 0.58,
        "user_4": 0.63,
        "user_5": 0.995,
        "user_6": 0.45,
    }

    sample_distributor = SampleDistributor(
        num_annotators=len(annotator_dict),
        time_available=10,
        annotation_rate=60,
        # num_samples=2160,
        double_proportion=1 / 3,
        re_proportion=1 / 2,
    )
    sample_distributor.set_project_distribution()
    print(sample_distributor)

    num_classes = 3
    df = generate_samples(sample_distributor, num_classes)
    sample_distributor.distribute_samples(df.copy(), "./data", all_reannotation=True)

    annotate_samples(annotator_dict, "./data", num_classes)
    annotations = concat_annotations("./data/annotations", len(annotator_dict))
    print(annotations)

    label_mapping = {0.0: 0, 1.0: 1, 2.0: 2}
    label_generator = EffiLabelGenerator(6, label_mapping)
    effiannos = Annotations(annotations, label_generator)
    effiannos.calculate_annotator_reliability()
    print(effiannos.get_reliability_dict())
    effiannos.display_annotator_graph()
