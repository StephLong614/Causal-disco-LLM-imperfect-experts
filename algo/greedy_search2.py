import operator
import numpy as np

get_cost = lambda p, size: np.log(p) - 0.5 * size

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
        


def greedy_search(gpt3_decision_probs, mec, undirected_edges, tol=0.501):
    decisions = []
    past_decision_score = -10000
    improvement = 1e-3
    while improvement > 0:
        decision_scores = {}
        
        for decision_potential in gpt3_decision_probs.keys():
            
            potential_new_mec = [dag for dag in mec if bool(decision_potential in dag)]
            resulting_decisions = get_decisions_from_mec(potential_new_mec, undirected_edges)
            dec = {'resulting_decisions': resulting_decisions,
                   'prob_wrong': 1 - np.prod([1-gpt3_decision_probs[dec] for dec in resulting_decisions]),
                   'mec_size': len(potential_new_mec),
                   'score': get_cost(p=np.prod([1-gpt3_decision_probs[dec] for dec in resulting_decisions]),
                                     size=len(potential_new_mec))}
            
            if (dec['score'] - past_decision_score) > 0:
                decision_scores[decision_potential] = dec
            
        decision_scores_ = sorted(decision_scores.items(), key=lambda item: item[1]['score'], reverse=True)

        if len(decision_scores_) > 0:
            decision_taken = decision_scores_[0]
        else:
            break

        print(decision_taken)
        decisions = decision_taken[1]['resulting_decisions']
        improvement = decision_taken[1]['score'] - past_decision_score
        past_decision_score = decision_taken[1]['score']
        mec = [dag for dag in mec if bool(decision_taken[0] in dag)]
    
    return mec, decisions

   