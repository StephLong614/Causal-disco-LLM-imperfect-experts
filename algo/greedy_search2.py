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
        
def greedy_search(observed_arcs, model, mec, undirected_edges, tol=0.501):
    decisions = []
    past_decision_score = -10000
    improvement = 1e-3
    while improvement > 0:
        decision_scores = {}
        
        for decision_potential in observed_arcs:
            
            potential_new_mec = [dag for dag in mec if bool(decision_potential in dag)]
            resulting_decisions = get_decisions_from_mec(potential_new_mec, undirected_edges)
            dec = {
                'resulting_decisions': resulting_decisions,
                'probability': model.posterior(observed_arcs, resulting_decisions),
                'mec_size': len(potential_new_mec)
            }

            dec['score'] = get_cost(p=dec['probability'], size=len(potential_new_mec))
            
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

# def greedy_search(observed_arcs, model, mec, undirected_edges, err_budget=0.501):
#     decisions = []
#     prob_correct = 1.

#     while undirected_edges:
#         decision_scores = {}
        
#         for candidate_arc in observed_arcs:
            
#             candidate_new_mec = [dag for dag in mec if candidate_arc in dag]
#             resulting_decisions = get_decisions_from_mec(candidate_new_mec, undirected_edges)

#             dec = {'resulting_decisions': resulting_decisions,
#                    'prob': model.posterior(resulting_decisions),
#                    'mec_size': len(candidate_new_mec),
#             }
            
#             if err_budget - 1 + dec['prob'] > 0:
#                 decision_scores[candidate_arc] = dec
            
#         if decision_scores:
#             decision_scores = sorted(decision_scores.items(), key=lambda item: item[1]['prob'], reverse=True)
#             decision_taken = decision_scores[0]
#         else:
#             break

#         print(decision_taken)
#         decisions = decision_taken[1]['resulting_decisions']
#         err_budget -= decision_taken[] 
#         past_decision_score = decision_taken[1]['score']
#         mec = [dag for dag in mec if decision_taken[0] in dag]

#         for dec in decisions:
#             if dec in undirected_edges:
#                 undirected_edges.remove(dec)
#             else:
#                 undirected_edges.remove((dec[1], dec[0]))
    
#     return mec, decisions

   