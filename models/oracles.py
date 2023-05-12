import numpy as np

class BaseOracle(object):
    """
    Base class for an expert that can be queried to obtain edge orientations

    """

    def __init__(self):
        super().__init__()
        
        self.likelihoods = dict()

    def decide(self, x1, x2):
        """
        Get edge orientation from expert

        Parameters:
        -----------
        x1: object
            A node in the graph
        x2: object
            Another node in the graph

        Returns:
        --------
        orientation: tuple
            The orientation of the edge (source, target) according to the expert.

        """
        raise NotImplementedError()
    
class EpsilonOracle(BaseOracle):
    """
    An expert that randomly lies about the true edge orientations with some probability.

    """

    def __init__(self, arcs, epsilon=0.05, random_state=np.random):
        """
        Constructs the expert

        Parameters:
        -----------
        arcs: list of tuples
            A list of the ground truth edge orientations
        epsilon: float, default: 0.05
            The probability with which the expert returns an incorrect orientation
        random_state: np.random.RandomState, default: np.random
            The random state to use for randomness

        """
        self.arcs = arcs
        self.epsilon = epsilon
        self.random_state = random_state
        super().__init__()

    def decide(self, x1, x2):
        """
        Get edge orientation from expert

        Parameters:
        -----------
        x1: object
            A node in the graph
        x2: object
            Another node in the graph

        Returns:
        --------
        orientation: tuple
            The orientation of the edge (source, target) according to the expert.
            Note that the expert lies with epsilon probability.

        """
        if (x1, x2) not in self.arcs and (x2, x1) not in self.arcs:
            raise ValueError(f"Edge {x1}--{x2} not in graph.")
        
        if (x1, x2) in self.arcs:
            true_edge, false_edge = (x1, x2), (x2, x1)
        else:
            true_edge, false_edge = (x2, x1), (x1, x2)
        
        if self.random_state.rand() < self.epsilon:
            
            self.likelihoods[false_edge] = self.epsilon
            
            return false_edge  
        
        else:
            
            self.likelihoods[true_edge] = 1 - self.epsilon
            
            return true_edge

    def decide_all(self):

        observations = []
        for arc in self.arcs:
            
            # Get decision by expert
            observations.append(self.decide(*arc))
        
        return observations