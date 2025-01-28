"""
Module to handle the preparation for anntotation.
This includes:
    * calculating number of samples to annotate
    * calculating the distribution of samples
"""

import warnings
from typing import Optional, List

import pandas as pd
from sympy import Eq, solve, symbols
from sympy.core.symbol import Symbol


def sample_without_replacement(
    df: pd.DataFrame, n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = min([len(df), n])
    sampled_df = df.sample(n)
    return df.drop(sampled_df.index.to_list()), sampled_df


def get_missing_var(variables: dict) -> Symbol:
    """Find the missing variable given a dict of variables.
       Exactly one variable should be None.

    Args:
        variables (dict): dict of variables {var: value}.

    Returns:
        Symbol: symbol of the missing variable using sympy.

    Raises:
        ValueError: if there are no missing variables or more
            than one missing variable.
    """
    missing_count = 0
    missing_variable = None

    for var, value in variables.items():
        if value is None:
            missing_variable = var
            missing_count += 1
        if missing_count > 1:
            raise ValueError("variables has more than one missing value.")

    if missing_variable is None:
        raise ValueError("variables does not have any missing values.")

    return missing_variable


class SampleDistributor:

    def __init__(
        self,
        annotators: Optional[List[str]] = None,
        num_annotators: Optional[int] = None,
        time_available: Optional[float] = None,
        annotation_rate: Optional[float] = None,
        num_samples: Optional[int] = None,
        double_proportion: Optional[float] = None,
        re_proportion: Optional[float] = None,
    ):
        if annotators is not None:
            if num_annotators != len(annotators):
                warnings.warn(f"Length of annotators and num_annotators do not match ({len(annotators)} != {num_annotators}). Setting num_annotators to {len(annotators)}")  # noqa
                num_annotators = len(annotators)
        self.get_variables(
            num_annotators,
            time_available,
            annotation_rate,
            num_samples,
            double_proportion,
            re_proportion,
        )
        if annotators is None:
            self.annotators = [f"user_{i}" for i in range(1, self.num_annotators + 1)]
        else:
            self.annotators = annotators

    def _assign_variables(self, variables: dict):
        """Assign class level variables from dict of symbolised
           variables (as defined in 'get_variables').

        Args:
            variables (dict): dict of variables to assign.
        """
        n, t, rho, k, d, r = symbols("n t rho k d r")
        self.num_annotators = variables.get(n)
        self.time_available = variables.get(t)
        self.annotation_rate = variables.get(rho)
        self.num_samples = int(variables.get(k))
        self.double_proportion = variables.get(d)
        self.re_proportion = variables.get(r)

    def get_variables(
        self,
        num_annotators: Optional[int] = None,
        time_available: Optional[float] = None,
        annotation_rate: Optional[float] = None,
        num_samples: Optional[int] = None,
        double_proportion: Optional[float] = None,
        re_proportion: Optional[float] = None,
    ):
        """Solves the annotation framework equation to find the missing
           variable. Only one of the available arguments should be ommitted.
        Args:
            num_annotators (int): number of annotators available [n].
            time_available (float): time available for each annotator
                (assuming they all have the same time available) [t].
            annotation_rate (float): expected rate of annotation per
                unit time (same unit as time_available) [rho].
            num_samples (int): number of desired samples [k].
            double_proportion (float): proportion of the whole dataset
                that should be double-annotated samples (0 <= n <= 1) [d].
            re_proportion (float): proportion of single-annotated samples
                that should be re-annotated (0 <= n <= 1) [r].
        """
        # define variable symbols
        n, t, rho, k, d, r = symbols("n t rho k d r")

        # define distribution equation
        equation = Eq(k, ((2 * d) + ((1 + r) * (1 - d))) ** (-1) * rho * t * n)

        # set vars
        variables = {
            n: num_annotators,
            t: time_available,
            rho: annotation_rate,
            k: num_samples,
            d: double_proportion,
            r: re_proportion,
        }

        # find missing var to solve for
        missing_variable = get_missing_var(variables)

        solution = solve(equation, missing_variable)

        # substitute values into solution
        variables[missing_variable] = solution[0].subs(
            {k: v for k, v in variables.items() if v is not None}
        )

        self._assign_variables(variables)

    def set_project_distribution(self):
        """Set project distributions once all values have been
        defined.
        """
        assert self.num_annotators is not None, "num_annotators must be set"  # noqa
        assert self.num_samples is not None, "num_samples must be set"  # noqa
        assert self.double_proportion is not None, "double_proportion must be set"  # noqa
        assert self.re_proportion is not None, "re_proportion must be set"  # noqa

        self.double_annotation_project = round(
            (self.double_proportion * self.num_samples) / (2 * self.num_annotators)  # noqa
        )
        self.single_annotation_project = round(
            ((1 - self.double_proportion) * self.num_samples) / self.num_annotators  # noqa
        )
        self.re_annotation_project = round(
            self.re_proportion * self.single_annotation_project
        )

    def create_example_distribution_df(self):
        """Create a simple DataFrame to test sample distribution."""
        assert self.num_samples is not None, "num_samples must be set"

        data = {"sample_number": range(1, self.num_samples * 2)}
        df = pd.DataFrame(data)

        return df

    def distribute_samples(
        self,
        df: pd.DataFrame,
        save_path: str = None,
        all_reannotation: bool = False,
    ):
        """Distribute samples based on sample distributor
           settings.

        Args:
            df (pd.DataFrame): dataframe containing samples with
                each row being a separate sample - using a copy
                is recommended.
            save_path (str): (Optional) If not None, dir path to save all data to.
                             If not supplied, a dict of allocations is returned.
                             Default None.
            all_reannotation (bool): whether re-annotations should be sampled
                from all the user's annotations rather than just single
                annotations. In this case, a double annotation project amount
                is sampled from all their annotations.
        """
        assert self.num_samples is not None, "num_samples must be set"
        assert self.num_annotators is not None, "num_annotators must be set"
        assert (
            self.double_annotation_project is not None
        ), "double_annotation_project must be set"
        assert self.double_proportion is not None, "double_proportion must be set"  # noqa
        assert (
            self.single_annotation_project is not None
        ), "single_annotation_project must be set"
        assert (
            self.re_annotation_project is not None
        ), "re_annotation_project must be set"

        if len(df) < self.num_samples:
            raise ValueError(
                f"DataFrame does not contain enough samples. len(df) [{len(df)}] < num_samples [{self.num_samples}]."  # noqa
            )

        # add sample_id to allow final dataset compilation
        df["sample_id"] = range(len(df))

        # create annotator dict
        annotations_dict = {user: [] for user in self.annotators}

        # TODO: maybe add some handling of save path?
        for (i, current_annotator) in enumerate(self.annotators):
            link_1_idx = (i+1) % self.num_annotators
            link_2_idx = (i+2) % self.num_annotators
            link_1_annotator = self.annotators[link_1_idx]
            link_2_annotator = self.annotators[link_2_idx]
            re_annotation_samples = None

            # single annotations
            if self.double_proportion < 1:
                df, single_samples = sample_without_replacement(
                    df, self.single_annotation_project
                )
                single_samples["is_reannotation"] = False

                annotations_dict[current_annotator].append(single_samples)

                if not all_reannotation:
                    re_annotation_samples = single_samples.sample(
                        self.re_annotation_project
                    )
                    re_annotation_samples["is_reannotation"] = True
                    annotations_dict[current_annotator].append(re_annotation_samples)  # noqa

            # double annotations
            if self.double_annotation_project > 0:
                df, first_double_samples = sample_without_replacement(
                    df, self.double_annotation_project
                )
                first_double_samples["is_reannotation"] = False

                annotations_dict[current_annotator].append(first_double_samples)  # noqa
                annotations_dict[link_1_annotator].append(first_double_samples)  # noqa

                df, second_double_samples = sample_without_replacement(
                    df, self.double_annotation_project
                )
                second_double_samples["is_reannotation"] = False

                annotations_dict[current_annotator].append(second_double_samples)  # noqa
                annotations_dict[link_2_annotator].append(second_double_samples)  # noqa

        if save_path is None:
            annotations_dict["left_over"] = df
            return annotations_dict

        for user, df_list in annotations_dict.items():
            # concat all user's dataframes
            user_df = pd.concat(df_list, ignore_index=True)
            # sample from all if not from singles
            if all_reannotation:
                re_annotation_samples = user_df.sample(self.double_annotation_project)  # noqa
                re_annotation_samples["is_reannotation"] = True
                user_df = pd.concat([user_df, re_annotation_samples], ignore_index=True)  # noqa
            # save df
            user_df.to_csv(f"{save_path}/{user}.csv", index=False)

        # save all left over samples
        df.to_csv(f"{save_path}/left_over.csv", index=False)

    def __str__(self):
        """String representation of sample distribution."""
        return (
            f"Variables:\n"
            f"num_annotators (n): {self.num_annotators}\n"
            f"time_available (t): {self.time_available}\n"
            f"annotation_rate (rho): {self.annotation_rate}\n"
            f"num_samples (k): {self.num_samples}\n"
            f"double_proportion (d): {self.double_proportion}\n"
            f"re_proportion (r): {self.re_proportion}\n"
            f"double_annotation_project: {self.double_annotation_project}\n"
            f"single_annotation_project: {self.single_annotation_project}\n"
            f"re_annotation_project: {self.re_annotation_project}"
        )

    def output_variables(self):
        """Output all variables."""
        print(self)
