import re

import numpy as np


def check_user_format(user_x):
    """Check that user is in the correct format of
    the string user_{some_number} or re_user_{some_number}.
    """
    return bool(re.match(r"(re_)?user_\d+", user_x))


def is_prob_label(x, num_classes):
    """Check that a value is a soft label (nd vector).

    Args:
        x: some value.
        num_classes: num classes for n-d vector.

    Returns:
        bool: whether x is a nd numpy vector.
    """
    return isinstance(x, np.ndarray) and x.shape == (num_classes,)


def headings_contain_prob_labels(df, heading_1, heading_2, num_classes=3):
    """Check that the given headings contain nothing but probability labels.

    Args:
        df (pd.DataFrame): the dataframe containing data to be worked with.
        heading_1 (str): first heading containing soft labels.
        heading_2 (str): second heading containing soft labels.

    Returns:
        bool: whether headings contain only soft labels.
    """
    checks = df[[heading_1, heading_2]].applymap(
        lambda x: is_prob_label(x, num_classes)
    )
    return checks.all(axis=None)


def retrieve_pair_annotations(df, user_x, user_y):
    """Get the subset of the dataframe annotated by users
       x and y.

    Args:
        df (pd.DataFrame): the whole dataset.
        user_x (str): name of the user in the form user_x.
        user_x (str): name of the user in the form user_y.

    Returns:
        pd.DataFrame: copy of the reduced subset containing
                      only samples annotated by both users.
    """
    if not check_user_format(user_x):
        raise ValueError(
            "User x parameters must be in the form user_x or re_user_x, where x is some number."
        )
    if not check_user_format(user_y):
        raise ValueError(
            "User y parameters must be in the form user_y or re_user_y, where y is some number."
        )

    return df[df[f"{user_x}_label"].notna() & df[f"{user_y}_label"].notna()].copy()


def csv_to_array(csv_string):
    """Convert csv string (such as value1,value2,...) to
       an array ["value1", "value2", ...]

    Args:
        csv_string (str): string to split and create an array from.

    Returns:
        list[str]: list of strings (if the value passed is a string),
            otherwise returns None.
    """
    if isinstance(csv_string, str):
        split_string = csv_string.split(",")
        return [x.strip() for x in split_string]
