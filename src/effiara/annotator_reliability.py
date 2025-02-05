import warnings
from itertools import combinations, product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from effiara.agreement import pairwise_agreement
from effiara.utils import retrieve_pair_annotations


class Annotations:
    """Class to hold all annotation information for the EffiARA annotation
    framework. Methods include inter- and intra- annotator agreement
    calculations, as well the overall reliability calculation and other
    utilities.

    Attributes:
        label_generator (effiara.LabelGenerator)
        annotators (list)
        num_annotators (int)
        label_mapping (dict)
        num_classes (int)
        agreement_metric (str)
        merge_labels (dict)
    """

    def __init__(
        self,
        df,
        label_generator,
        agreement_metric="krippendorff",
        merge_labels=None,
    ):
        """
        Args:
            label_generator (effiara.LabelGenerator)
            agreement_metric (str)
            merge_labels (dict): Optional.
        """
        # set instance variables
        self.label_generator = label_generator
        self.annotators = label_generator.annotators
        self.num_annotators = label_generator.num_annotators
        self.label_mapping = label_generator.label_mapping
        self.num_classes = label_generator.num_classes
        self.agreement_metric = agreement_metric
        self.merge_labels = merge_labels

        # set in self.calculate_inter_annotator_agreement
        self.overall_inter_annotator_agreement = np.nan

        # load dataset
        self.df = df.copy()

        # merge labels
        self.replace_labels()

        # generate user soft labels
        self.df = self.label_generator.add_annotation_prob_labels(self.df)

        # calculate agreements
        self.G = self.init_annotator_graph()
        self.calculate_intra_annotator_agreement()
        self.calculate_inter_annotator_agreement()
        self.calculate_annotator_reliability()

    def replace_labels(self):
        """Merge labels. Uses find and replace so do not switch labels e.g.
        {"misinfo": ["debunk"], "debunk": ["misinfo", "other"]}.
        """
        if not self.merge_labels:
            return

        for replacement, to_replace in self.merge_labels.items():
            for label in to_replace:
                for user in self.annotators:
                    label_col = f"{user}_label"
                    re_label_col = "re_" + label_col
                    secondary_col = f"{user}_secondary"
                    re_secondary_col = "re_" + secondary_col

                    # find and replace in each col
                    self.df[label_col] = self.df[label_col].replace(label, replacement)  # noqa
                    self.df[secondary_col] = self.df[secondary_col].replace(
                        label, replacement
                    )
                    self.df[re_label_col] = self.df[re_label_col].replace(
                        label, replacement
                    )
                    self.df[re_secondary_col] = self.df[re_secondary_col].replace(  # noqa
                        label, replacement
                    )

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
                "Mean value must be greater than zero, high agreement/reliability will become low and vice versa.")  # noqa

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
                "Mean value must be greater than zero, high agreement/reliability will become low and vice versa.")  # noqa

        for node in self.G.nodes():
            self.G.nodes[node][property] /= avg

    def calculate_inter_annotator_agreement(self, threshold=30):
        """Calculate the inter-annotator agreement between each
        pair of annotators. Each agreement value will be
        represented on the edges of the graph between nodes
        that are representative of each annotator.

        Args:
            threshold (int): threshold number of samples required
                for a link to be made between two annotators.

        """
        inter_annotator_agreement_scores = {}
        pairs = combinations(self.annotators, 2)
        for (current_annotator, link_annotator) in pairs:
            # TODO: optimise use of pair df rather than generate twice
            pair_df = retrieve_pair_annotations(
                    self.df, current_annotator, link_annotator)
            if len(pair_df) >= threshold:
                pair = (current_annotator, link_annotator)
                inter_annotator_agreement_scores[pair] = (
                    pairwise_agreement(
                        self.df,
                        current_annotator,
                        link_annotator,
                        self.label_mapping,
                        num_classes=self.num_classes,
                        metric=self.agreement_metric,
                    )
                )

        # add all agreement scores to the graph
        for users, score in inter_annotator_agreement_scores.items():
            self.G.add_edge(users[0], users[1], agreement=score)

        self.overall_inter_annotator_agreement = np.mean(
            list(inter_annotator_agreement_scores.values())
        )

    def calculate_intra_annotator_agreement(self):
        """Calculate intra-annotator agreement."""
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
                )
            except KeyError:
                warnings.warn(
                    "Key error for calculating intra-annotator agreement. Setting all intra-annotator agreement values to 1.")  # noqa
                self.G.nodes[user]["intra_agreement"] = 1
            except Exception as e:
                self.G.nodes[user]["intra_agreement"] = 1
                print(e)

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

    def calculate_annotator_reliability(
            self, alpha=0.5, beta=0.5, epsilon=0.001):
        """Recursively calculate annotator reliability, using
           intra-annotator agreement, inter-annotator agreement,
           or a mixture, controlled by the alpha and beta parameters.
           Alpha and Beta must sum to 1.0.

        Args:
            alpha (float): Default 0.5. Value between 0 and 1 controlling weight of intra-annotator agreement.  # noqa
            beta (float): Default 0.5. Value between 0 and 1, controlling weight of inter-annotator agreement.  # noqa
            epsilon (float): Default 0.001. Controls the maximum change from the last iteration to indicate convergence.  # noqa
        """
        if alpha + beta != 1:
            raise ValueError("Alpha and Beta must sum to 1.0.")

        if alpha < 0 or alpha > 1 or beta < 0 or beta > 1:
            raise ValueError("Alpha and beta values must be between 0 and 1.")

        # keep updating until convergence
        max_change = np.inf
        while abs(max_change) > epsilon:
            print("Running iteration.")
            previous_reliabilties = {
                    node: data["reliability"]
                    for (node, data) in self.G.nodes(data=True)}

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
                    abs(self.G.nodes[node]["reliability"] - previous_reliabilties[node])  # noqa
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
        return {node: self.G.nodes[node]["reliability"]
                for node in self.G.nodes()}

    def display_annotator_graph(self, legend=False):
        """Display the annotation graph."""
        plt.figure(figsize=(12, 12))
        pos = nx.circular_layout(self.G, scale=0.9)

        node_size = 3000
        nx.draw_networkx_nodes(self.G, pos, node_size=node_size)
        nx.draw_networkx_edges(self.G, pos)
        # Get the usernames.
        labels = {node: node.split('_', maxsplit=1)[-1]
                  for node in self.G.nodes()}
        nx.draw_networkx_labels(
            self.G, pos, labels=labels, font_color="white", font_size=24
        )

        # add inter-annotator agreement to edges
        edge_labels = {(u, v): f"{d['agreement']:.3f}"
                       for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(
                self.G, pos, edge_labels=edge_labels, font_size=24)

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
            reliability_scores = {node: data["reliability"]
                                  for (node, data) in self.G.nodes(data=True)}
            texts = [f"{node}: {score:.3f}"
                     for (node, score) in reliability_scores.items()]
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
            annotators: list = None,
            other_annotators: list = None):
        """Plot a heatmap of agreement metric values for the annotators.

        If both annotators and other_annotators are specifed, compares
        users in annotators to those in other_annotators. Otherwise,
        compare all project annotators to each other. 

        Args:
            annotators (list): Optional.
            other_annotators (list): Optional.
        """
        mat = nx.to_numpy_array(self.G, weight="agreement")
        # Put intra-agreements on the diagonal
        intras = nx.get_node_attributes(self.G, "intra_agreement")
        intras = np.array(list(intras.values()))
        mat[np.diag_indices(mat.shape[0])] = intras
        agreements = self.G.nodes(data="avg_inter_agreement")
        if annotators is not None and other_annotators is not None:
            matrows = [i for (i, user) in enumerate(self.annotators)
                       if user in annotators]
            matcols = [i for (i, user) in enumerate(self.annotators)
                       if user in other_annotators]
            # If we're comparing two sets of annotators,
            # slice the agreement matrix.
            mat = mat[matrows][:, matcols]
            agreements = [(name, agree) for (name, agree) in agreements
                          if name in annotators]


        sorted_by_agreement = sorted(enumerate(agreements),
                                     key=lambda n: n[1][1], reverse=True)
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
            mat[np.triu_indices(mat.shape[0], k=1)] = np.nan
            xlabs = ylabs = sorted_users
        else:
            xlabs = other_annotators
            ylabs = sorted_users
        sns.heatmap(mat, annot=True, fmt='.3f',
                    xticklabels=xlabs, yticklabels=ylabs)
        plt.show()

    def __str__(self):
        return_string = ""
        for node, attrs in self.G.nodes(data=True):
            return_string += f"Node {node} has the following attributes:\n"
            for attr, value in attrs.items():
                return_string += f"  {attr}: {value}\n"
            return_string += "\n"
        return return_string
