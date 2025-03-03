import numpy as np


def is_prob_label(x, num_classes):
    """Check that a value is a soft label (nd vector).

    Args:
        x (np.ndarray): a vector of values.
        num_classes (int): the number of classes.

    Returns:
        bool: whether x is a nd numpy vector.
    """
    is_array = isinstance(x, np.ndarray)
    if not is_array:
        return False
    if np.any(x < 0):
        return False
    is_correct_shape = x.shape == (num_classes,)
    sums_to_one = np.isclose(np.sum(x), 1)
    return is_correct_shape and sums_to_one


def headings_contain_prob_labels(df, heading_1, heading_2, num_classes=3):
    """Check that the given headings contain nothing but probability labels.

    Args:
        df (pd.DataFrame): the dataframe containing data to be worked with.
        heading_1 (str): first heading containing soft labels.
        heading_2 (str): second heading containing soft labels.

    Returns:
        bool: whether headings contain only soft labels.
    """
    checks = df[[heading_1, heading_2]].map(lambda x: is_prob_label(x, num_classes))
    return checks.all(axis=None)


def retrieve_pair_annotations(df, user_x, user_y, suffix="_label"):
    """Get the subset of the dataframe annotated by users
       x and y.

    Args:
        df (pd.DataFrame): the whole dataset.
        user_x (str): name of the user in the form user_x.
        user_x (str): name of the user in the form user_y.
        suffix (str): suffix added to usernames, used to
            find all shared samples.

    Returns:
        pd.DataFrame: copy of the reduced subset containing
                      only samples annotated by both users.
    """
    condition1 = df[f"{user_x}{suffix}"].notna()
    condition2 = df[f"{user_y}{suffix}"].notna()
    return df[condition1 & condition2].copy()


# TODO: this really needs changing to csv_to_list
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
