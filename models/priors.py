import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

from causaldag import DAG
from collections import defaultdict

class MECPrior(object):
    def __init__(self, cpdag):
        """
        A class for calculating the prior probability of edge orientations in a Markov Equivalence
        Class. The probability is determined by enumerating all DAGs in the MEC and counting edge
        orientation frequencies. Only edges that are undirected in the CPDAG are considered.

        Parameters
        ----------
        cpdag : causaldag.PDAG
            A CPDAG object representing the Markov equivalence class of DAGs.

        """

        self._cpdag = cpdag

        # Get all orientations for undirected CPDAG edges
        self.unoriented_edges = list(
            chain(*[[(x1, x2), (x2, x1)] for x1, x2 in cpdag.edges])
        )

        # Assign an ID to each orientation
        self._uedge_idx = dict(
            zip(self.unoriented_edges, range(len(self.unoriented_edges)))
        )

        # Prior table (n_orientations x n_dags_in_mec)
        self._prior = np.zeros(
            (len(self.unoriented_edges), len(self._cpdag.all_dags()))
        )

        # Get occurence of each edge in all DAGs of the MEC
        for i, dag in enumerate(self._cpdag.all_dags()):
            for edge, idx in self._uedge_idx.items():
                if edge in dag:
                    self._prior[idx, i] = 1

        super().__init__()

    def __call__(self, edges):
        """
        Compute the probability of the given set of edges in the equivalence class.

        Parameters
        ----------
        edges : list of tuples
            A list of edges, each represented as a tuple (source, target), where source and target
            are the nodes in the graph.

        Returns
        -------
        float
            The probability of the given set of edges occurring in the equivalence class of DAGs,
            computed as the product of the occurrence of each edge in all possible DAGs, divided
            by the total number of possible DAGs in the equivalence class.

        """
        edge_idx = [self._uedge_idx[e] for e in edges]
        return (
            np.vstack([self._prior[self._uedge_idx[e]] for e in edges])
            .prod(axis=0)
            .sum()
            / self._prior.shape[1]
        )