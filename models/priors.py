from abc import abstractmethod

import numpy as np

from itertools import chain, product

class Prior(object):

    def __init__(self, cpdag):
        """
        A class for calculating the prior probability of arcs in a Markov Equivalence Class. 
        The probability corresponds to a uniform distribution over DAGs in the MEC. 
        Hence, only edges that are undirected in the CPDAG are considered/useful.

        Parameters
        ----------
        cpdag : causaldag.PDAG
            A CPDAG object representing the Markov equivalence class of DAGs.

        """

        self._cpdag = cpdag

        # Get all orientations for undirected CPDAG edges
        self.arcs = list(
            chain(*[[(x1, x2), (x2, x1)] for x1, x2 in cpdag.edges])
        )

        # Assign an ID to each orientation
        self._arc_idx = dict(
            zip(self.arcs, range(len(self.arcs)))
        )

        # Prior table (n_orientations x n_dags_in_mec)
        self._prior = np.zeros(
            (len(self.arcs), self.support_size())
        )

        # Get occurence of each edge in all DAGs of the MEC
        for i, dag in enumerate(self._cpdag.all_dags()):
            for edge, idx in self._arc_idx.items():
                if edge in dag:
                    self._prior[idx, i] = 1
    
    @abstractmethod
    def enumerate(self):
        """
        All possible complete orientations of edges and their probabilities. 

        Returns
        -------
        generator
            each item is a tuple: (probability, list of arcs)
        """
        pass

    @abstractmethod
    def support_size(self):
        """
        Return the number of possible combinations of arcs (with non-zero probability)
        """
        pass
    
    @abstractmethod
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
        pass

class IndependentPrior(Prior):

    def __init__(self, cpdag, weights="uniform"):

        super(IndependentPrior, self).__init__(cpdag)

        self._arc_weights = np.ones(len(self.arcs))

        if weights == "uniform":
            self._arc_weights /= 2 # exactly 50% chances for an arc of being oriented either way
        
        elif weights == "occurences":
            for arc, idx in self._arc_idx.items():
                self._arc_weights[idx] = self._prior(arc)
            
        else:
            raise NotImplementedError

    def __call__(self, arcs):
        """
        Compute the probability of the given set of arcs in the equivalence class, supposing independence.

        Parameters
        ----------
        arcs : list of tuples
            A list of arcs, each represented as a tuple (source, target), where source and target
            are the nodes in the graph.

        Returns
        -------
        float
            The probability of the given set of arcs as the product of the probability of each arc.

        """
        return np.prod([self._arc_weights[self._arc_idx[e]] for e in arcs])
    
    def enumerate(self):
        """
        All possible complete orientations of CPDAG's edges. 

        Returns
        -------
        generator
            each item is a list of arcs.
        """
        for complete_orientation in product(*[(self.arcs[i], self.arcs[i+1]) for i in range(0, len(self.arcs), 2)]):

            yield self(complete_orientation), complete_orientation

    def support_size(self):
        return 2 ** (len(self.arcs) // 2)
    
class MECPrior(Prior):

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
        return (
            np.vstack([self._prior[self._arc_idx[e]] for e in arcs])
            .prod(axis=0)
            .sum()
            / self._prior.shape[1]
        )

    def enumerate(self):
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

            # uniform distribution
            yield 1. / self.support_size(), complete_orientation
    
    def support_size(self):
        return len(self._cpdag.all_dags())