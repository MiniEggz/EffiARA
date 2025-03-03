import pytest

from effiara.utils import csv_to_array


# test valid strings
@pytest.mark.parametrize(
    "csv_string, expected_output",
    [
        ("value1,value2,value3", ["value1", "value2", "value3"]),
        (
            "apple, banana, cherry",
            ["apple", "banana", "cherry"],
        ),  # spaces should be stripped
        (
            "  spaced , values , here  ",
            ["spaced", "values", "here"],
        ),  # extra spaces should be removed
        ("one,two,three,four", ["one", "two", "three", "four"]),
        ("singlevalue", ["singlevalue"]),  # single value should still be a list
        ("1,2,3,4,5", ["1", "2", "3", "4", "5"]),  # numeric values as strings
        (
            "value-with-hyphen,value_with_underscore",
            ["value-with-hyphen", "value_with_underscore"],
        ),
    ],
)
def test_valid_csv_strings(csv_string, expected_output):
    assert (
        csv_to_array(csv_string) == expected_output
    ), f"Expected {expected_output}, got {csv_to_array(csv_string)}"


# test non-string inputs
@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        123,
        4.56,
        ["already", "a", "list"],  # lists should not be processed
        {"key": "value"},  # dicts should not be processed
        (1, 2, 3),  # tuples should not be processed
        set(["a", "b", "c"]),  # sets should not be processed
    ],
)
def test_invalid_inputs(invalid_input):
    assert (
        csv_to_array(invalid_input) is None
    ), f"Expected None for input {invalid_input}"


# TODO: not sure whether this is desired - will get caught at later point
# empty and malformed strings
@pytest.mark.parametrize(
    "csv_string, expected_output",
    [
        ("", [""]),  # an empty string should return a list with an empty string
        (",,,", ["", "", "", ""]),  # just commas should return empty elements
        (" , , , ", ["", "", "", ""]),  # commas with spaces should behave the same
        (
            ",value1,,value2,",
            ["", "value1", "", "value2", ""],
        ),  # leading/trailing commas
    ],
)
def test_edge_cases(csv_string, expected_output):
    assert (
        csv_to_array(csv_string) == expected_output
    ), f"Expected {expected_output}, got {csv_to_array(csv_string)}"
