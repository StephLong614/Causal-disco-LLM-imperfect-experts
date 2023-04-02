import numpy as np

def get_prior(undirected_edges, mec):
    prior_probs = {}
    for edge in undirected_edges:
      node_i = edge[0]
      node_j = edge[1]
      
      # how often in the mec does j->i
      edges = np.array([(node_j, node_i) in g for g in mec])
      prior_probs[(node_i, node_j)] = 1-np.mean(edges)
      prior_probs[(node_j, node_i)] = np.mean(edges)
    
    return prior_probs


def get_posterior(prior, likelihood):
  """
  P(B|A) = P(A|B)P(B)/P(A)
         = P(A|B)P(B)/(P(A|B=1)P(B=1) + P(A|B=0)P(B=0))
  """
  return likelihood