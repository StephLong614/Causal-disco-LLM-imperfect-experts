import operator
import numpy as np

from utils.dag_utils import get_decisions_from_mec

def greedy_search(observed_arcs, model, mec, undirected_edges, tol=0.501):
    decisions = []
    p_correct = 1.
    while (p_correct > tol) and (len(mec) > 1):
        decision_scores = {}
        
        for decision_potential in observed_arcs:
            
            potential_new_mec = [dag for dag in mec if decision_potential in dag]
            resulting_decisions = get_decisions_from_mec(potential_new_mec, undirected_edges)
            dec = {
                'resulting_decisions': resulting_decisions,
                'probability': 1 - model.posterior(observed_arcs, resulting_decisions),
                'mec_size': len(potential_new_mec)
            }
            
            if (dec['probability'] > tol) and (len(potential_new_mec) > 0):
                decision_scores[decision_potential] = dec
            
        decision_scores = sorted(decision_scores.items(), key=lambda item: item[1]['mec_size'], reverse=False)
        
        decision_taken = decision_scores[0]
        decisions = decision_taken[1]['resulting_decisions']
        mec = [dag for dag in mec if decision_taken[0] in dag]
        p_correct *= decision_taken[1]['probability']
    
    return mec, decisions, p_correct

   