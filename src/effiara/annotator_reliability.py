import warnings
from itertools import combinations, product
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from effiara.agreement import pairwise_agreement
from effiara.label_generators import DefaultLabelGenerator, LabelGenerator
from effiara.utils import retrieve_pair_annotations


class Annotations:
    """Class to hold all annotation information for the EffiARA annotation
    framework. Methods include inter- and intra- annotator agreement
    calculations, as well the overall reliability calculation and other
    utilities.

    Attributes:
        label_generator (effiara.LabelGenerator): label generator to create
            individual annotation labels and soft/hard aggregations.
        annotators (list): list of annotator names.
        num_annotators (int): number of annotators
        label_mapping (dict): label mapping of what is in the dataframe to what
            should be used for agreement/training.
        num_classes (int): number of classes.
        agreement_metric (str): agreement metric to be used.
        agreement_suffix (str): label suffix to get the agreement from (such as
            "_label" as the default).
        agreement_type (str): type of agreement (e.g. nominal, ordinal).
        merge_labels (dict): dict of labels to merge.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_generator: Optional[LabelGenerator] = None,
        agreement_metric: str = "krippendorff",
        agreement_suffix: str = "_label",
        agreement_type: str = "nominal",
        overlap_threshold: int = 15,
        merge_labels: Optional[dict] = None,
        reliability_alpha: float = 0.5,
        reannotations=False,
    ):
        """
        Args:
            df
            label_generator (effiara.LabelGenerator)
            annotation_col_suffixes (List[str])
            agreement_metric (str)
            agreement_suffix (str)
            agreement_type (str)
            overlap_threshold (str)
            merge_labels (dict): Optional.
            reliability_alpha (float): control strength of intra-annotator
                agreement in reliability calculations.
        """
        # set instance variables
        self.df = df.copy()

        # ensure label generator set
        if label_generator is None:
            label_generator = DefaultLabelGenerator.from_annotations(self.df)

        self.label_generator = label_generator
        assert self.label_generator is not None
        self.annotators = label_generator.annotators
        self.num_annotators = label_generator.num_annotators
        self.label_mapping = label_generator.label_mapping
        self.num_classes = label_generator.num_classes
        self.annotation_col_suffixes = self.label_generator.label_suffixes

        self.agreement_metric = agreement_metric
        self.agreement_suffix = agreement_suffix
        self.agreement_type = agreement_type

        self.overlap_threshold = overlap_threshold

        self.merge_labels = merge_labels

        self.reliability_alpha = reliability_alpha

        # TODO: add some check for reannotations
        self.reannotations = reannotations  # TODO: replace w/ function to look for re_

        # set in self.calculate_inter_annotator_agreement
        self.overall_inter_annotator_agreement = np.nan

        # merge labels
        self.replace_labels()

        # generate annotation labels
        self.df = self.label_generator.add_annotation_prob_labels(self.df)

        # calculate agreements
        self.G = self.init_annotator_graph()
        self.calculate_intra_annotator_agreement()
        self.calculate_inter_annotator_agreement()
        self.calculate_annotator_reliability(alpha=self.reliability_alpha)

    def __getitem__(self, users):
        if isinstance(users, str):
            # Just one user specified
            item = self.G.nodes()[users]
        else:
            if not isinstance(users, (tuple, list)):
                raise KeyError(str(users))
            if len(users) > 2:
                raise ValueError(f"__getitem__ takes one or two users.")
            item = self.G.edges()[users]
        return item

    def replace_labels(self):
        """Merge labels. Uses find and replace so do not switch labels e.g.
        {"misinfo": ["debunk"], "debunk": ["misinfo", "other"]}.
        """
        if not self.merge_labels:
            return

        # TODO: refactor to remove as many loops
        for replacement, to_replace in self.merge_labels.items():
            for label in to_replace:
                for user in self.annotators:
                    for suffix in self.annotation_col_suffixes:
                        label_col = f"{user}{suffix}"
                        self.df[label_col] = self.df[label_col].replace(
                            label, replacement
                        )  # noqa

                        if self.reannotations:
                            re_label_col = "re_" + label_col
                            self.df[re_label_col] = self.df[re_label_col].replace(
                                label, replacement
                            )

    # TODO: rename the label generator names
    def generate_final_labels_and_sample_weights(self):
        """Generate the final labels and sample weights for the dataframe."""
        self.df = self.label_generator.add_sample_prob_labels(
            self.df, self.get_reliability_dict()
        )

        self.df = self.label_generator.add_sample_hard_labels(self.df)

    def init_annotator_graph(self):
        """Initialise the annotator graph with an initial reliability of 1.
        This means each annotator will initially be weighted equally.
        """
        G = nx.Graph()
        for user in self.annotators:
            G.add_node(user, reliability=1)
        return G

    def normalise_edge_property(self, property):
        """Normalise an edge property to have a mean of 1.

        Args:
            property (str): the name of the edge property to normalise.
        """
        total = sum(edge[property] for _, _, edge in self.G.edges(data=True))
        num_edges = self.G.number_of_edges()

        avg = total / num_edges
        if avg < 0:
            raise ValueError(
                "Mean value must be greater than zero, high agreement/reliability will become low and vice versa."
            )  # noqa

        for _, _, edge in self.G.edges(data=True):
            edge[property] /= avg

    def normalise_node_property(self, property):
        """Normalise a node property to have a mean of 1.

        Args:
            property (str): the name of the node property to normalise.
        """
        total = sum(node[property] for _, node in self.G.nodes(data=True))
        num_nodes = self.G.number_of_nodes()

        avg = total / num_nodes
        if avg < 0:
            raise ValueError(
                "Mean value must be greater than zero, high agreement/reliability will become low and vice versa."
            )  # noqa

        for node in self.G.nodes():
            self.G.nodes[node][property] /= avg

    def calculate_inter_annotator_agreement(self):
        """Calculate the inter-annotator agreement between each
        pair of annotators. Each agreement value will be
        represented on the edges of the graph between nodes
        that are representative of each annotator.
        """
        inter_annotator_agreement_scores = {}
        pairs = combinations(self.annotators, 2)
        for current_annotator, link_annotator in pairs:
            pair_df = retrieve_pair_annotations(
                self.df, current_annotator, link_annotator
            )
            if len(pair_df) >= self.overlap_threshold:
                pair = (current_annotator, link_annotator)
                inter_annotator_agreement_scores[pair] = pairwise_agreement(
                    self.df,
                    current_annotator,
                    link_annotator,
                    self.label_mapping,
                    num_classes=self.num_classes,
                    metric=self.agreement_metric,
                    agreement_type=self.agreement_type,
                    label_suffix=self.agreement_suffix
                )

        # add all agreement scores to the graph
        for users, score in inter_annotator_agreement_scores.items():
            self.G.add_edge(users[0], users[1], agreement=score)

        # TODO: maybe add alternative way of anntotator agreement?
        # i.e. Krippendorff for all annotations if individual annotator
        # doesn't matter
        self.overall_inter_annotator_agreement = np.mean(
            list(inter_annotator_agreement_scores.values())
        )

    # MINOR CHANGE -- add self.reannotations filter
    # TODO: maybe allow to change default value (np.nan for example)
    def calculate_intra_annotator_agreement(self):
        """Calculate intra-annotator agreement."""
        if self.reannotations:
            for user in self.annotators:
                re_user = f"re_{user}"
                try:
                    self.G.nodes[user]["intra_agreement"] = pairwise_agreement(  # noqa
                        self.df,
                        user,
                        re_user,
                        self.label_mapping,
                        num_classes=self.num_classes,
                        metric=self.agreement_metric,
                        agreement_type=self.agreement_type,
                        label_suffix=self.agreement_suffix
                    )
                except KeyError:
                    warnings.warn(
                        "Key error for calculating intra-annotator agreement. Setting all intra-annotator agreement values to 1."
                    )  # noqa
                    self.G.nodes[user]["intra_agreement"] = 1
                except Exception as e:
                    self.G.nodes[user]["intra_agreement"] = 1
                    print(e)
        else:
            for user in self.annotators:
                self.G.nodes[user]["intra_agreement"] = 1

    def calculate_avg_inter_annotator_agreement(self):
        """Calculate each annotator's average agreement using
        using a weighted average from the annotators around
        them. The average is weighted by the overall reliability
        score of each annotator.
        """
        for node in self.G.nodes():
            edges = self.G.edges(node, data=True)
            # get weighted avg agreement
            weighted_agreement_sum = 0
            weights_sum = 0
            for _, target, edge in edges:
                weight = self.G.nodes[target]["reliability"]
                weights_sum += weight
                weighted_agreement_sum += weight * edge["agreement"]
            self.G.nodes[node]["avg_inter_agreement"] = (
                weighted_agreement_sum / weights_sum if weights_sum else 0
            )

    def reset_annotator_reliability(self):
        for _, node in self.G.nodes(data=True):
            node["reliability"] = 1.0

    def calculate_annotator_reliability(self, alpha=0.5, epsilon=0.001):
        """Recursively calculate annotator reliability, using
           intra-annotator agreement, inter-annotator agreement,
           or a mixture, controlled by the alpha and beta parameters.
           Alpha and Beta must sum to 1.0.

        Args:
            alpha (float): Default 0.5. Value between 0 and 1 controlling weight of intra-annotator agreement.  # noqa
            beta (float): Default 0.5. Value between 0 and 1, controlling weight of inter-annotator agreement.  # noqa
            epsilon (float): Default 0.001. Controls the maximum change from the last iteration to indicate convergence.  # noqa
        """
        if alpha > 1 or alpha < 0:
            raise ValueError("Alpha must be set such that 0 <= alpha <= 1.")

        beta = 1 - alpha

        # set to 1 here to allow for re-calculation with different alpha
        self.reset_annotator_reliability()

        # keep updating until convergence
        max_change = np.inf
        while abs(max_change) > epsilon:
            # print("Running iteration.")
            previous_reliabilties = {
                node: data["reliability"] for (node, data) in self.G.nodes(data=True)
            }

            # calculate the new inter annotator agreement scores
            self.calculate_avg_inter_annotator_agreement()

            # update reliability
            for _, node in self.G.nodes(data=True):
                intra = node["intra_agreement"]
                inter = node["avg_inter_agreement"]
                rel = float(alpha * intra + beta * inter)
                node["reliability"] = rel
            self.normalise_node_property("reliability")

            # find largest change as a marker
            max_change = max(
                [
                    abs(
                        self.G.nodes[node]["reliability"] - previous_reliabilties[node]
                    )  # noqa
                    for node in self.G.nodes()
                ]
            )

    def get_user_reliability(self, username):
        """Get the reliability of a given annotator.

        Args:
            username (str): username of the annotator.

        Returns:
            float: reliability score of the annotator.
        """
        return self.G.nodes[username]["reliability"]

    def get_reliability_dict(self):
        """Get a dictionary of reliability scores per username.

        Returns:
            dict: dictionary of key=username, value=reliability.
        """
        return {node: self.G.nodes[node]["reliability"] for node in self.G.nodes()}

    def display_annotator_graph(self, legend=False):
        """Display the annotation graph."""
        plt.figure(figsize=(12, 12))
        pos = nx.circular_layout(self.G, scale=0.9)

        node_size = 3000
        nx.draw_networkx_nodes(self.G, pos, node_size=node_size)
        nx.draw_networkx_edges(self.G, pos)
        # Get the usernames.
        labels = {node: node.split("_", maxsplit=1)[-1] for node in self.G.nodes()}
        nx.draw_networkx_labels(
            self.G, pos, labels=labels, font_color="white", font_size=24
        )

        # add inter-annotator agreement to edges
        edge_labels = {
            (u, v): f"{d['agreement']:.3f}" for u, v, d in self.G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=24)

        # adjust text pos for intra-annotator agreement
        for node, (x, y) in pos.items():
            if x == 0:
                align = "center"
                if y > 0:
                    y_offset = 0.15
                else:
                    y_offset = -0.15
            elif y == 0:
                align = "center"
                y_offset = 0 if x > 0 else -0.15
            elif x > 0:
                align = "left"
                y_offset = 0.15 if y > 0 else -0.15
            else:
                align = "right"
                y_offset = 0.15 if y > 0 else -0.15

            plt.text(
                x,
                y + y_offset,
                s=f"{self.G.nodes[node]['intra_agreement']:.3f}",
                horizontalalignment=align,
                verticalalignment="center",
                fontdict={"color": "black", "size": 24},
            )

        # legend for reliability
        if legend:
            reliability_scores = {
                node: data["reliability"] for (node, data) in self.G.nodes(data=True)
            }
            texts = [
                f"{node}: {score:.3f}" for (node, score) in reliability_scores.items()
            ]
            reliability_text = "Reliability:\n\n" + "\n".join(texts)
            plt.text(
                0.05,
                0.95,
                reliability_text,
                transform=plt.gca().transAxes,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=12,
                color="black",
            )

        # plot
        plt.axis("off")
        plt.show()

    def display_agreement_heatmap(
        self,
        annotators: Optional[list] = None,
        other_annotators: Optional[list] = None,
        display_upper=False,
    ):
        """Plot a heatmap of agreement metric values for the annotators.

        If both annotators and other_annotators are specifed, compares
        users in annotators to those in other_annotators. Otherwise,
        compare all project annotators to each other.

        Args:
            annotators (list): Optional.
            other_annotators (list): Optional.

        Returns:
            np.ndarray: A matrix of the data displayed on the graph.
            List[str]: List of annotators in the order of the matrix rows.
        """
        mat = nx.to_numpy_array(self.G, weight="agreement")
        # Put intra-agreements on the diagonal
        intras = nx.get_node_attributes(self.G, "intra_agreement")
        intras = np.array(list(intras.values()))
        mat[np.diag_indices(mat.shape[0])] = intras
        agreements = self.G.nodes(data="avg_inter_agreement")
        if annotators is not None and other_annotators is not None:
            matrows = [
                i for (i, user) in enumerate(self.annotators) if user in annotators
            ]
            matcols = [
                i
                for (i, user) in enumerate(self.annotators)
                if user in other_annotators
            ]
            # If we're comparing two sets of annotators,
            # slice the agreement matrix.
            mat = mat[matrows][:, matcols]
            agreements = zip(annotators, np.mean(mat, axis=1))

        sorted_by_agreement = sorted(
            enumerate(agreements), key=lambda n: n[1][1], reverse=True
        )
        ordered_row_idxs = [i for (i, _) in sorted_by_agreement]
        mat = mat[ordered_row_idxs]

        # We now have two possible cases.
        #  1) annotators and other_annotators == None: We're comparing
        #     each annotator to each other. In this case we'll display
        #     only the lower triangle of the agreement heatmap as the
        #     the upper triangle will be identical to the lower.
        #  2) otherwise, we're comparing two possibly distinct sets of
        #     annotators, so we display the full matrix, with rows and
        #     columns sliced according to the annotators specified.
        sorted_users = [user for (i, (user, agree)) in sorted_by_agreement]
        if other_annotators is None:
            mat = mat[:, ordered_row_idxs]
            # Don't display upper triangle, since its redundant.
            if not display_upper:
                mat[np.triu_indices(mat.shape[0], k=1)] = np.nan
            xlabs = ylabs = sorted_users
        else:
            xlabs = [user for user in self.annotators if user in other_annotators]
            ylabs = sorted_users
        sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=xlabs, yticklabels=ylabs)
        plt.show()
        return mat, sorted_users

    def __str__(self):
        return_string = ""
        for node, attrs in self.G.nodes(data=True):
            return_string += f"Node {node} has the following attributes:\n"
            for attr, value in attrs.items():
                return_string += f"  {attr}: {value}\n"
            return_string += "\n"
        return return_string
