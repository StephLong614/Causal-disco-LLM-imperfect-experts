from collections import OrderedDict
import numpy as np
import operator

from utils.language_models import gpt3_scoring

def global_scoring(gpt3_decision_probs, prior_probs, mec, undirected_edges, tol=0.101):

    all_scores = []
    for dag in mec:
        score = 0
        denom = 0
        for edge in dag:
            # the graph is node_i -> node_j
            node_i = edge[0]
            node_j = edge[1]
            # only score edges that are not yet determined
            if ((node_i, node_j) in undirected_edges) or ((node_j, node_i) in undirected_edges):
                error = gpt3_decision_probs[(node_i, node_j)]
                # if score is less than 50\% the LM believes that 
                # node_i -> node_j, thus we increase the score of the graph
                # under the model by 1 otherwise 0
                edge_score = 1 if (error < 0.5) else 0
                score += edge_score
                denom += 1
                
        all_scores.append(score/denom)
    
    # only keep the 1/2 of the scoring graphs
    top_indices = np.argsort(all_scores)[len(all_scores)//2:]
    mec = [mec[i] for i in top_indices]
    return mec, {}