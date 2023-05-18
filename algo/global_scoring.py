import numpy as np

from utils.dag_utils import get_mec

def global_scoring(observed_arcs, model, cpdag, undirected_edges, **kwargs):

    all_scores = []

    mec = get_mec(cpdag)
    for dag in mec:
        score, denom = 0, 0
        
        # only score edges that are not yet determined
        for edge in (set(dag) - cpdag.arcs):
            # score += int(edge in observed_arcs)
            score += int(model([edge], [edge]) > 0.5)
            denom += 1
                
        all_scores.append(score/denom)
    top_indices = np.argwhere(all_scores == np.amax(all_scores)).flatten()

    mec = [mec[i] for i in top_indices]
    scores = [all_scores[i] for i in top_indices]

    return mec, dict(), np.mean(scores)