import operator
import numpy as np


def get_decisions_from_mec(mec, undirected_edges):
    decisions = []
    
    for edge in undirected_edges:
        node_i = edge[0]
        node_j = edge[1]
        i_j = np.sum([((node_i, node_j) in dag) for dag in mec])
        j_i = np.sum([((node_j, node_i) in dag) for dag in mec])
        # if i_j and j_i we don't have to make a decision
        if not (i_j and j_i):
            if i_j:
                decisions.append((node_i, node_j))
            else:
                decisions.append((node_j, node_i))
                
    return decisions
        


def greedy_search(gpt3_decision_probs, mec, undirected_edges, tol=0.101):
    decisions = []
    eps = 0.
    while eps < tol:
        decision_scores = {}
        
        for decision_potential in gpt3_decision_probs.keys():
            
            potential_new_mec = [dag for dag in mec if bool(decision_potential in dag)]
            resulting_decisions = get_decisions_from_mec(potential_new_mec, undirected_edges)
            dec = {'resulting_decisions': resulting_decisions,
                   'prob_wrong': 1 - np.prod([gpt3_decision_probs[dec] for dec in resulting_decisions]),
                   'mec_size': len(potential_new_mec)}
            
            if dec['prob_wrong'] <= tol:
                decision_scores[decision_potential] = dec
            
        decision_scores = sorted(decision_scores.items(), key=lambda item: item[1]['mec_size'], reverse=False)
        
        decision_taken = decision_scores[0]
        print(decision_taken)
        decisions = decision_taken[1]['resulting_decisions']
        mec = [dag for dag in mec if bool(decision_taken[0] in dag)]
        eps += decision_taken[1]['prob_wrong']
    
    return mec, decisions

   