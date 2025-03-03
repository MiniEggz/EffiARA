import re

import pandas as pd
import pytest

from effiara.label_generators import DefaultLabelGenerator


# test DefaultLabelGenerator instantiation (numeric labels)
def test_default_label_generator_instantiation_numeric():
    generator = DefaultLabelGenerator(["user1", "user2"], {0: 0, 1: 1, 2: 2})

    assert isinstance(generator, DefaultLabelGenerator)
    assert generator.num_annotators == 2
    assert generator.num_classes == 3
    assert generator.label_mapping == {0: 0, 1: 1, 2: 2}
    assert generator.label_suffixes == ["_label"]


# test defaultlabelgenerator instantiation (string labels)
def test_default_label_generator_instantiation_string():
    generator = DefaultLabelGenerator(
        ["user1", "user2"], {"misinfo": 0, "debunk": 1, "other": 2}
    )

    assert isinstance(generator, DefaultLabelGenerator)
    assert generator.num_annotators == 2
    assert generator.num_classes == 3
    assert generator.label_mapping == {"misinfo": 0, "debunk": 1, "other": 2}


# TODO: add tests for different labels in reannotation columns
# test from_annotations() with numeric labels
def test_default_label_generator_from_annotations_numeric():
    df = pd.DataFrame(
        {
            "user_1_label": [0, 1, 2, 1],
            "user_2_label": [0, 1, 2, 2],
            "true_label": [0, 1, 2, 1],
        }
    )

    generator = DefaultLabelGenerator.from_annotations(df)

    assert isinstance(generator, DefaultLabelGenerator)
    assert generator.num_annotators == 2
    assert generator.num_classes == 3
    assert generator.label_mapping == {0: 0, 1: 1, 2: 2}


# test from_annotations() with string labels
def test_default_label_generator_from_annotations_string():
    df = pd.DataFrame(
        {
            "user1_label": ["misinfo", "debunk", "other", "misinfo"],
            "user2_label": ["debunk", "misinfo", "other", "debunk"],
            "true_label": ["misinfo", "debunk", "other", "misinfo"],
        }
    )

    generator = DefaultLabelGenerator.from_annotations(df)

    assert isinstance(generator, DefaultLabelGenerator)
    assert generator.num_annotators == 2
    assert set(generator.label_mapping.keys()) == {"misinfo", "debunk", "other"}
    assert list(generator.label_mapping.values()) == list(range(3))
    assert len(generator.label_mapping) == 3


# test from_annotations() with no label columns (should raise ValueError)
def test_default_label_generator_from_annotations_no_labels():
    df = pd.DataFrame({"true_label": ["misinfo", "debunk", "other"]})

    with pytest.raises(
        ValueError, match=re.escape("No *_label columns found in annotations!")
    ):
        DefaultLabelGenerator.from_annotations(df)


# test from_annotations() with num_classes less than found labels
def test_default_label_generator_from_annotations_fewer_num_classes():
    df = pd.DataFrame(
        {
            "user1_label": ["misinfo", "debunk", "other", "misinfo"],
            "user2_label": ["debunk", "misinfo", "other", "debunk"],
        }
    )

    with pytest.raises(
        ValueError,
        match="num_classes.*less than the number of classes found in annotations",
    ):
        DefaultLabelGenerator.from_annotations(df, num_classes=2)


# TODO: change functionality so this test fails
# ideally, want to tell the user when they don't have all the labels
# test from_annotations() with extra num_classes (Handling Placeholders)
def test_default_label_generator_from_annotations_extra_num_classes():
    df = pd.DataFrame(
        {
            "user1_label": ["misinfo", "debunk", "other", "misinfo"],
            "user2_label": ["debunk", "misinfo", "other", "debunk"],
        }
    )

    generator = DefaultLabelGenerator.from_annotations(df, num_classes=5)

    assert generator.num_classes == 5
    assert len(generator.label_mapping) == 5  # Ensure placeholders were added


# check that default methods return the annotations as they are
def test_default_label_generator_methods():
    df = pd.DataFrame(
        {
            "user1_label": ["misinfo", "debunk", "other"],
            "user2_label": ["debunk", "misinfo", "other"],
        }
    )
    generator = DefaultLabelGenerator(
        ["user1", "user2"], {"misinfo": 0, "debunk": 1, "other": 2}
    )

    assert generator.add_annotation_prob_labels(df).equals(df)
    assert generator.add_sample_prob_labels(df, {}).equals(df)
    assert generator.add_sample_hard_labels(df).equals(df)
