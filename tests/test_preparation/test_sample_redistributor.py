import pandas as pd
import pytest

from effiara.preparation import SampleDistributor
from effiara.preparation import SampleRedistributor


# class initialisation
def test_sample_redistributor_from_sample_distributor():
    sd = SampleDistributor(
        annotators=[f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        # num_samples=2160,  # 2160
        double_proportion=1.0 / 3,
        re_proportion=0.5,
    )
    sr = SampleRedistributor.from_sample_distributor(sd)

    assert isinstance(sr, SampleRedistributor)
    assert sr.annotators == sd.annotators
    assert sr.num_samples == sd.num_samples
    assert sr.double_proportion == 0.0
    assert sr.re_proportion == 0.0
    assert sr.annotation_rate == sd.annotation_rate


def test_sample_redistributor_init():
    sr = SampleRedistributor(
        annotators=[f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
        # need one missing
        # time_available=10,
        annotation_rate=60,
        num_samples=2160,  # 2160
        double_proportion=0.0,
        re_proportion=0.0,
    )

    assert isinstance(sr, SampleRedistributor)
    assert sr.annotators == ["user_1", "user_2", "user_3", "user_4", "user_5", "user_6"]
    assert sr.num_samples == 2160
    assert sr.double_proportion == 0.0
    assert sr.re_proportion == 0.0
    assert sr.annotation_rate == 60


# TODO: add successful sample redistribution
# use data generator to generate the initial annotations


# unsupported double should raise assertion error
def test_unsupported_double():
    sr = SampleRedistributor(
        annotators=[f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=None,  # 2160
        double_proportion=1.0 / 3,
        re_proportion=0.0,
    )

    sr.set_project_distribution()

    with pytest.raises(AssertionError, match="Double annotation not yet supported"):
        sr.distribute_samples(
            pd.DataFrame({"user_1_label": [None] * 10, "user_2_label": [None] * 10})
        )


# unsupported reannotation should raise assertion error
def test_unsupported_reannotation():
    sr = SampleRedistributor(
        annotators=[f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=None,  # 2160
        double_proportion=0.0,
        re_proportion=0.5,
    )

    sr.set_project_distribution()

    with pytest.raises(AssertionError, match="Reannotation not yet supported"):
        sr.distribute_samples(
            pd.DataFrame({"user_1_label": [None] * 10, "user_2_label": [None] * 10})
        )


# test missing annotations in DataFrame
def test_missing_annotations():
    sr = SampleRedistributor(
        annotators=[f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=None,  # 2160
        double_proportion=0.0,
        re_proportion=0.0,
    )
    sr.set_project_distribution()

    with pytest.raises(ValueError, match="No annotations found in dataframe!"):
        sr.distribute_samples(pd.DataFrame({"data": range(10000)}))


# test insufficient sample count
def test_insufficient_sample_count():
    df = pd.DataFrame({"user_1_label": [None] * 2, "user_2_label": [None] * 2})

    sr = SampleRedistributor(
        annotators=["user_1", "user_2"],
        num_samples=5,
        annotation_rate=1,
        double_proportion=0.0,
        re_proportion=0.0,
    )
    sr.set_project_distribution()

    with pytest.raises(ValueError, match="DataFrame does not contain enough samples"):
        sr.distribute_samples(df)


# test warning for not all samples allocated
def test_unallocated_warning():
    df = pd.DataFrame(
        {"user_1_label": ["A", None, None], "user_2_label": ["B", "C", None]}
    )

    sr = SampleRedistributor(
        annotators=["user_1", "user_2"],
        num_samples=2,
        annotation_rate=1,
        double_proportion=0.0,
        re_proportion=0.0,
    )
    sr.set_project_distribution()

    with pytest.warns(UserWarning, match="Not all examples were able to be allocated"):
        allocations = sr.distribute_samples(df.copy())
        assert "left_over" in allocations
        assert len(allocations["left_over"]) > 0
