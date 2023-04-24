import numpy as np

from itertools import chain

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
        self.edges = list(
            chain(*[[(x1, x2), (x2, x1)] for x1, x2 in cpdag.edges])
        )

        # Assign an ID to each orientation
        self._edge_idx = dict(
            zip(self.edges, range(len(self.edges)))
        )

        # Prior table (n_orientations x n_dags_in_mec)
        self._prior = np.zeros(
            (len(self.edges), self.mec_size())
        )

        # Get occurence of each edge in all DAGs of the MEC
        for i, dag in enumerate(self._cpdag.all_dags()):
            for edge, idx in self._edge_idx.items():
                if edge in dag:
                    self._prior[idx, i] = 1

        super().__init__()

    def __call__(self, arcs):
        """
        Compute the probability of the given set of arcs in the equivalence class.

        Parameters
        ----------
        arcs : list of tuples
            A list of arcs, each represented as a tuple (source, target), where source and target
            are the nodes in the graph.

        Returns
        -------
        float
            The probability of the given set of arcs occurring in the equivalence class of DAGs,
            computed as the product of the occurrence of each edge in all possible DAGs, divided
            by the total number of possible DAGs in the equivalence class.

        """
        edge_idx = [self._edge_idx[e] for e in arcs]
        return (
            np.vstack([self._prior[self._edge_idx[e]] for e in arcs])
            .prod(axis=0)
            .sum()
            / self._prior.shape[1]
        )

    def get_complete_orientations(self):
        """
        All possible complete orientations of CPDAG's edges. 

        Returns
        -------
        generator
            each item is a list of arcs corresponding to one DAG in the MEC.
        """
        for dag in self._cpdag.all_dags():
            complete_orientation = list()

            for x1, x2 in dag:
                if self._cpdag.has_edge(x1, x2):
                    complete_orientation.append((x1, x2))

            yield complete_orientation
    
    def mec_size(self):
        return len(self._cpdag.all_dags())