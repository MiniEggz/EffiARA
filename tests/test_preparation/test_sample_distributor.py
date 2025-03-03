import math
import warnings
from itertools import combinations

import pandas as pd
import pytest

from effiara.preparation import SampleDistributor


# test valid sample distributor init
@pytest.mark.parametrize(
    "missing_attr",
    [
        "annotators",
        "time_available",
        "annotation_rate",
        "num_samples",
        "double_proportion",
        "re_proportion",
    ],
)
def test_sample_distributor_initialization(missing_attr):
    kwargs = {
        "annotators": [f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
        "time_available": 10,
        "annotation_rate": 60,
        "num_samples": 2160,  # 2160
        "double_proportion": 1.0 / 3,
        "re_proportion": 0.5,
    }
    kwargs.pop(missing_attr)
    distributor = SampleDistributor(**kwargs)

    assert distributor.num_annotators == 6
    assert distributor.annotators == [
        "user_1",
        "user_2",
        "user_3",
        "user_4",
        "user_5",
        "user_6",
    ]
    # check outputs are numeric
    assert isinstance(distributor.time_available, (int, float))
    assert isinstance(distributor.annotation_rate, (int, float))
    assert isinstance(distributor.double_proportion, float)
    assert isinstance(distributor.re_proportion, float)
    assert isinstance(distributor.num_samples, int)

    assert math.isclose(distributor.time_available, 10)
    assert math.isclose(distributor.annotation_rate, 60)
    assert distributor.num_samples == 2160
    assert math.isclose(distributor.double_proportion, 1.0 / 3)
    assert math.isclose(distributor.re_proportion, 0.5)


# test when all values have already been set
def test_all_values_set():

    with pytest.raises(ValueError, match="variables does not have any missing values."):
        SampleDistributor(
            annotators=[f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
            time_available=10,
            annotation_rate=60,
            num_samples=2160,  # 2160
            double_proportion=1.0 / 3,
            re_proportion=0.5,
        )


# check when more than one value is missing
@pytest.mark.parametrize(
    "missing_attrs",
    list(
        combinations(
            [
                "annotators",
                "time_available",
                "annotation_rate",
                "num_samples",
                "double_proportion",
                "re_proportion",
            ],
            2,
        )
    ),
)
def test_missing_multiple_parameters(missing_attrs):
    kwargs = {
        "annotators": [f"user_{i}" for i in range(1, 7)],
        "time_available": 10,
        "annotation_rate": 60,
        "num_samples": 2160,
        "double_proportion": 1.0 / 3,
        "re_proportion": 0.5,
    }
    for missing_attr in missing_attrs:
        kwargs.pop(missing_attr)

    with pytest.raises(ValueError, match=f"variables has more than one missing value"):
        SampleDistributor(**kwargs)


# test that project distribution variables are set
def test_set_project_distribution():
    distributor = SampleDistributor(
        annotators=None,  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=2160,  # 2160
        double_proportion=1.0 / 3,
        re_proportion=0.5,
    )
    distributor.set_project_distribution()

    assert distributor.double_annotation_project == 60
    assert distributor.single_annotation_project == 240
    assert distributor.re_annotation_project == 120


# test example df creation
def test_create_example_distribution_df():
    distributor = SampleDistributor(
        annotators=None,  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=2160,  # 2160
        double_proportion=1.0 / 3,
        re_proportion=0.5,
    )
    df = distributor.create_example_distribution_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4320
    assert "sample_number" in df.columns


# test successful sample distribution
@pytest.mark.parametrize(
    "missing_attr",
    [
        "annotators",
        "time_available",
        "annotation_rate",
        "num_samples",
        "double_proportion",
        "re_proportion",
    ],
)
def test_distribute_samples(missing_attr):
    kwargs = {
        "annotators": [f"user_{i}" for i in range(1, 7)],  # 6 annotators 1-6
        "time_available": 10,
        "annotation_rate": 60,
        "num_samples": 2160,  # 2160
        "double_proportion": 1.0 / 3,
        "re_proportion": 0.5,
    }
    kwargs.pop(missing_attr)
    distributor = SampleDistributor(**kwargs)

    df = pd.DataFrame({"data": range(3000)})

    distributor.set_project_distribution()
    allocations = distributor.distribute_samples(df.copy())

    for k, v in allocations.items():
        if k != "left_over":
            assert len(v) == 600  # len(annotators) * time_available
        else:
            assert len(v) == 840


# TODO: add functionality for solving all_reannotation properly


# check ValueError raised for lack of samples
def test_distribute_samples_insufficient_data():
    distributor = SampleDistributor(
        annotators=None,  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=2160,  # 2160
        double_proportion=1.0 / 3,
        re_proportion=0.5,
    )
    df = pd.DataFrame({"data": range(50)})  # Only 50 samples

    distributor.set_project_distribution()

    with pytest.raises(ValueError, match="DataFrame does not contain enough samples"):
        distributor.distribute_samples(df)


# test string representation
def test_sample_distributor_str():
    distributor = SampleDistributor(
        annotators=None,  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=2160,  # 2160
        double_proportion=1.0 / 3,
        re_proportion=0.5,
    )
    distributor.set_project_distribution()

    output_str = str(distributor)
    assert "num_annotators (n): 6" in output_str
    assert "time_available (t): 10" in output_str
    assert "annotation_rate (rho): 60" in output_str
    assert "num_samples (k): 2160" in output_str
    assert f"double_proportion (d): {1.0 / 3}" in output_str
    assert "re_proportion (r): 0.5" in output_str
    assert "double_annotation_project: 60" in output_str
    assert "single_annotation_project: 240" in output_str
    assert "re_annotation_project: 120" in output_str


# test output_variables
def test_output_variables(capsys):
    distributor = SampleDistributor(
        annotators=None,  # 6 annotators 1-6
        time_available=10,
        annotation_rate=60,
        num_samples=2160,  # 2160
        double_proportion=1.0 / 3,
        re_proportion=0.5,
    )
    distributor.set_project_distribution()
    distributor.output_variables()

    captured = capsys.readouterr()
    expected_output = str(distributor)

    print(captured)
    print(expected_output)

    assert expected_output.strip() == captured.out.strip()
