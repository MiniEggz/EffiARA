import math

import networkx as nx
import pandas as pd
import pytest

from effiara.annotator_reliability import Annotations


# test initialisation of annotations
def test_annotations_initialization():
    df = pd.DataFrame(
        {
            "user_1_label": [0, 1, 2, 1],
            "user_2_label": [0, 1, 2, None],
            "user_3_label": [None, 1, 2, 0],
            "true_label": [0, 1, 2, 1],
        }
    )

    annotations = Annotations(df)

    assert isinstance(annotations, Annotations)
    assert annotations.annotators == ["user_1", "user_2", "user_3"]
    assert annotations.num_annotators == 3
    assert annotations.agreement_metric == "krippendorff"
    assert annotations.agreement_suffix == "_label"


# test replacing labels
def test_replace_labels():
    df = pd.DataFrame(
        {
            "user_1_label": ["misinfo", "debunk", "other"],
            "user_2_label": ["debunk", "misinfo", "other"],
            "true_label": ["misinfo", "debunk", "other"],
        }
    )

    merge_labels = {"other": ["other", "debunk"]}

    annotations = Annotations(df, merge_labels=merge_labels)
    assert annotations.annotators == ["user_1", "user_2"]
    annotations.replace_labels()

    assert (annotations.df["user_1_label"] == ["misinfo", "other", "other"]).all()
    assert (annotations.df["user_2_label"] == ["other", "misinfo", "other"]).all()


# test graph initialisation
def test_init_annotator_graph():
    df = pd.DataFrame({"user_1_label": [0, 1, 2], "user_2_label": [1, 1, 2]})
    annotations = Annotations(df)

    G = annotations.init_annotator_graph()
    assert isinstance(G, nx.Graph)
    assert set(G.nodes) == {"user_1", "user_2"}
    assert all(G.nodes[node]["reliability"] == 1 for node in G.nodes)


# test inter-annotator agreement
@pytest.mark.parametrize(
    "df, expected_value",
    [
        (
            pd.DataFrame(
                {"user_1_label": [0, 1, 2, 1], "user_2_label": [0, 1, 2, None]}
            ),
            1.0,
        ),
        (
            pd.DataFrame(
                {"user_1_label": [1, 2, 2, 1], "user_2_label": [0, 1, 0, None]}
            ),
            -0.25,
        ),
    ],
)
def test_calculate_inter_annotator_agreement(df, expected_value):
    annotations = Annotations(df, overlap_threshold=3)
    annotations.calculate_inter_annotator_agreement()

    assert math.isclose(annotations.overall_inter_annotator_agreement, expected_value)
    assert ("user_1", "user_2") in annotations.G.edges


# test intra-annotator agreement calculation
@pytest.mark.parametrize(
    "df, expected_value",
    [
        (
            pd.DataFrame(
                {"user_1_label": [0, 1, 2, 1], "re_user_1_label": [0, 1, 2, None]}
            ),
            1.0,
        ),
        (
            pd.DataFrame(
                {"user_1_label": [1, 1, 2, 1, 2], "re_user_1_label": [0, 1, 2, None, 2]}
            ),
            0.631578947368421,
        ),
        (
            pd.DataFrame(
                {
                    "user_1_label": [1, 2, 2, 1, None],
                    "re_user_1_label": [0, 1, 0, None, None],
                    "user_2_label": [1, 1, 2, 2, 2],
                    "re_user_2_label": [1, 1, 2, 2, 2],
                }  # add user 2 to not raise lower than 0 error for normalise
            ),
            -0.25,
        ),
    ],
)
def test_calculate_intra_annotator_agreement(df, expected_value):
    annotations = Annotations(df, reannotations=True, overlap_threshold=3)
    annotations.calculate_intra_annotator_agreement()

    assert math.isclose(
        annotations.G.nodes["user_1"]["intra_agreement"], expected_value
    )


# test reliability calculation
@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_calculate_annotator_reliability(alpha):
    df = pd.DataFrame(
        {
            "user_1_label": [0, 1, 2, 1],
            "re_user_1_label": [0, 1, 2, None],
            "user_2_label": [0, 1, 2, None],
            "re_user_2_label": [0, 0, 0, None],
            "user_3_label": [0, 1, 0, 1],
            "re_user_3_label": [None, 0, 1, 0],
        }
    )
    annotations = Annotations(
        df, overlap_threshold=3, reannotations=True, reliability_alpha=alpha
    )

    assert annotations.get_user_reliability(
        "user_1"
    ) > annotations.get_user_reliability("user_2")
    assert annotations.get_user_reliability(
        "user_2"
    ) > annotations.get_user_reliability("user_3")


# test invalid alpha
def test_calculate_annotator_reliability_invalid_alpha():
    df = pd.DataFrame({"user1_label": [0, 1, 2]})
    annotations = Annotations(df)

    with pytest.raises(
        ValueError, match="Alpha must be set such that 0 <= alpha <= 1."
    ):
        annotations.calculate_annotator_reliability(alpha=1.5)


# test dict retrieval
def test_get_reliability_dict():
    df = pd.DataFrame(
        {
            "user_1_label": [0, 1, 2, 1],
            "re_user_1_label": [0, 1, 2, None],
            "user_2_label": [0, 1, 2, None],
            "re_user_2_label": [0, 0, 0, None],
            "user_3_label": [0, 1, 0, 1],
            "re_user_3_label": [None, 0, 1, 0],
        }
    )
    annotations = Annotations(df, overlap_threshold=3, reannotations=True)

    reliability_dict = annotations.get_reliability_dict()
    assert isinstance(reliability_dict, dict)
    assert set(reliability_dict.keys()) == {"user_1", "user_2", "user_3"}
    assert all(isinstance(r, float) for r in reliability_dict.values())
