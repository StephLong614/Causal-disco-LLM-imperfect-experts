import numpy as np

from utils.dag_utils import get_decisions_from_mec, get_mec

def greedy_search_mec_size(observed_arcs, model, cpdag, undirected_edges, tol=0.501):
    decisions = []
    p_correct = 1.
    mec = get_mec(cpdag)

    while (p_correct > 1 - tol) and (len(mec) > 1):
        decision_scores_ = {}
        
        for decision_potential in observed_arcs:
            
            potential_new_mec = [dag for dag in mec if decision_potential in dag]
            resulting_decisions = get_decisions_from_mec(potential_new_mec, undirected_edges)
            dec = {
                'resulting_decisions': resulting_decisions,
                'probability': model(observed_arcs, resulting_decisions),
                'mec_size': len(potential_new_mec)
            }
            
            if (p_correct * dec['probability'] > 1 - tol) and (len(potential_new_mec) > 0):
                decision_scores_[decision_potential] = dec
                    
        if len(decision_scores_) > 0:
            decision_scores_ = sorted(decision_scores_.items(), key=lambda item: item[1]['mec_size'], reverse=False)
            decision_taken = decision_scores_[0]
        else:
            break
    
        decision_taken = decision_scores_[0]
        decisions = decision_taken[1]['resulting_decisions']
        mec = [dag for dag in mec if decision_taken[0] in dag]
        p_correct *= decision_taken[1]['probability']
    
    return mec, decisions, p_correct

def greedy_search_confidence(observed_arcs, model, cpdag, undirected_edges, tol=0.501):
    decisions = []
    p_correct = 1.
    mec = get_mec(cpdag)

    while (p_correct > 1 - tol) and (len(mec) > 1):
        decision_scores_ = {}
        
        for decision_potential in observed_arcs:
            
            potential_new_mec = [dag for dag in mec if decision_potential in dag]
            resulting_decisions = get_decisions_from_mec(potential_new_mec, undirected_edges)
            dec = {
                'resulting_decisions': resulting_decisions,
                'probability': model(observed_arcs, resulting_decisions),
                'mec_size': len(potential_new_mec)
            }
            
            if (p_correct * dec['probability'] > 1 - tol) and (len(potential_new_mec) > 0):
                decision_scores_[decision_potential] = dec
                    
        if len(decision_scores_) > 0:
            decision_scores_ = sorted(decision_scores_.items(), key=lambda item: item[1]['probability'], reverse=True)
            decision_taken = decision_scores_[0]
        else:
            break
    
        decision_taken = decision_scores_[0]
        decisions = decision_taken[1]['resulting_decisions']
        mec = [dag for dag in mec if decision_taken[0] in dag]
        p_correct *= decision_taken[1]['probability']
    
    return mec, decisions, p_correct

get_cost = lambda p, size: np.log(p) - 0.5 * size
        
def greedy_search_bic(observed_arcs, model, cpdag, undirected_edges, **kwargs):
    decisions = []

    p_correct = 1.
    past_decision_score = -10000
    improvement = 1e-3
    mec = get_mec(cpdag)

    while improvement > 0:
        decision_scores = {}
        
        for decision_potential in observed_arcs:
            
            potential_new_mec = [dag for dag in mec if bool(decision_potential in dag)]
            resulting_decisions = get_decisions_from_mec(potential_new_mec, undirected_edges)
            dec = {
                'resulting_decisions': resulting_decisions,
                'probability': model(observed_arcs, resulting_decisions),
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

        decisions = decision_taken[1]['resulting_decisions']
        improvement = decision_taken[1]['score'] - past_decision_score
        past_decision_score = decision_taken[1]['score']
        mec = [dag for dag in mec if bool(decision_taken[0] in dag)]
        p_correct *= decision_taken[1]['probability']

    return mec, decisions, p_correct