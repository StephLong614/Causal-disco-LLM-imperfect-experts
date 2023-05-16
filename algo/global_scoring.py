from collections import OrderedDict
import numpy as np
import operator

from utils.language_models import gpt3_scoring
from utils.dag_utils import get_mec

is_in_undirected_edges = lambda node_i, node_j, undirected_edges: ((node_i, node_j) in undirected_edges) or ((node_j, node_i) in undirected_edges)

def global_scoring(observed_arcs, model, cpdag, undirected_edges, **kwargs):

    all_scores = []
    top_indices = []

    mec = get_mec(cpdag)
    for dag in mec:
        score = 0
        denom = 0
        for edge in dag:
            # the graph is node_i -> node_j
            node_i = edge[0]
            node_j = edge[1]
            # only score edges that are not yet determined
            if is_in_undirected_edges(node_i, node_j, undirected_edges):
                prob = model(observed_arcs, [edge])
                # if prob is more than 50\% the LM believes that 
                # node_i -> node_j, thus we increase the score of the graph
                # under the model by 1 otherwise 0
                edge_score = 1 if (prob > 0.5) else 0 #1=yes; was formerly less than
                score += edge_score
                denom += 1
                
        all_scores.append(score/denom)
    

    top_indices = np.argwhere(all_scores == np.amax(all_scores)).flatten()

    mec = [mec[i] for i in top_indices]
    scores = [all_scores[i] for i in top_indices]

    return mec, dict(), np.mean(scores)