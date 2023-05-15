from itertools import product
import numpy as np

class NoisyExpert(object):
    def __init__(self, prior, likelihoods):
        """
        A class for calculating the posterior probability of edge orientations (arcs) in a Markov 
        Equivalence Class, based on the assumption that when queried for an edge the expert returns
        an orientation that depends only on the true orientation of that edge. 
        Only edges that are undirected in the CPDAG are considered.

        Parameters
        ----------
        
        prior : models.priors.Prior
            Provides the prior probability of any subset of arcs
        
        likelihoods : dict {arc: arc_likelihood}
            Likelihood of each observed arc
        """

        self._prior = prior
        self._likelihoods = likelihoods
    
    def _partition_function(self, obs_arcs):

        ans = 0.

        for prob, true_arcs in self._prior.enumerate():

            # p(obs|true)
            ans += self.likelihood(obs_arcs, true_arcs) * prob

        return ans
    
    # likelihood of a set of orientations is factorizable P(O|E)
    def likelihood(self, obs_arcs, true_arcs):
        # TODO: check lists contain exactly one orientation of each edge

        ans = 1.        
        for (x1, x2) in obs_arcs:

            if (x1, x2) in true_arcs:
                ans *= self._likelihoods[(x1, x2)]
            else:
                ans *= 1 - self._likelihoods[(x1, x2)]
            
        return ans

    # posterior probability of subset of possible arcs
    def posterior(self, obs_arcs, true_arcs):
        
        ans = 0.

        # if an edge is not oriented in the decision, need to marginalize over it
        margin_edges = list()
        for (x1, x2) in obs_arcs:

            if ((x1, x2) not in true_arcs) and ((x2, x1) not in true_arcs):
                margin_edges.append(((x1, x2), (x2, x1)))
        
        if len(margin_edges) > 0:
            
            # Cartesian product over possible orientations of edges
            for arcs in product(*margin_edges):
                ans += self.likelihood(obs_arcs, list(arcs) + true_arcs) * self._prior(list(arcs) + true_arcs)
        
        else:
            ans = self.likelihood(obs_arcs, true_arcs) * self._prior(true_arcs)

        return ans / self._partition_function(obs_arcs)