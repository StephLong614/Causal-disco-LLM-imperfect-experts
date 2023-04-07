import numpy as np

def get_prior(undirected_edges, mec, uniform_prior):
    prior_probs = {}
    for edge in undirected_edges:
      node_i = edge[0]
      node_j = edge[1]
      
      # how often in the mec does j->i
      edges = np.array([(node_j, node_i) in g for g in mec])
      prior_probs[(node_i, node_j)] = 0.5 if uniform_prior else (1 - np.mean(edges))
      prior_probs[(node_j, node_i)] = 0.5 if uniform_prior else np.mean(edges)
    
    return prior_probs


def get_posterior(prior, likelihood):
  """
  P(B|A) = P(A|B)P(B)/P(A)
         = P(A|B)P(B)/(P(A|B=1)P(B=1) + P(A|B=0)P(B=0))
  """
  posterior = {}
  for edge in likelihood:
    num = likelihood[edge] * prior[edge]
    anti_edge = (edge[1], edge[0])
    denom = likelihood[edge] * prior[edge] + likelihood[anti_edge] * prior[anti_edge]
    posterior[edge] = num/denom
  return posterior