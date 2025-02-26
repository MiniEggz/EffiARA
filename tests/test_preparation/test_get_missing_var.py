import pytest

from effiara.preparation import get_missing_var


# valid test cases
@pytest.mark.parametrize(
    "variables, expected_output",
    [
        ({"x": 5, "y": None, "z": 10}, "y"),
        ({"a": None, "b": 2, "c": 3}, "a"),
        ({"m": 4, "n": 8, "p": None}, "p"),
    ],
)
def test_valid_get_missing_var(variables, expected_output):
    assert (
        get_missing_var(variables) == expected_output
    ), f"Expected {expected_output}, got {get_missing_var(variables)}"


# multiple missing values
@pytest.mark.parametrize(
    "variables", [{"x": None, "y": None, "z": 10}, {"a": None, "b": None, "c": None}]
)
def test_multiple_missing_vars(variables):
    with pytest.raises(ValueError, match="variables has more than one missing value."):
        get_missing_var(variables)


# no missing values
@pytest.mark.parametrize(
    "variables",
    [
        {"x": 1, "y": 2, "z": 3},
        {"a": 4, "b": 5, "c": 6},
        {},
    ],
)
def test_no_missing_vars(variables):
    with pytest.raises(ValueError, match="variables does not have any missing values."):
        get_missing_var(variables)


# non-dict input should fail
@pytest.mark.parametrize("invalid_input", [None, [], "string", 42, set(["x", "y"])])
def test_invalid_input_type(invalid_input):
    with pytest.raises(TypeError, match="variables must be of type 'dict'."):
        get_missing_var(invalid_input)
