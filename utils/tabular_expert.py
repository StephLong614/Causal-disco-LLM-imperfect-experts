import random


def get_tabular_probs(undirected_edges, codebook, true_G, epsilon=0.0):
    print('epsilon ', epsilon)
    
    decision_probs = {}
    edge_list = list(true_G.edges())

    for edge in undirected_edges:
        # get the correct ordering of the edge
        node_i, node_j = edge if (edge in edge_list) else (edge[1], edge[0])
        
        if random.random() < epsilon:
            # make an error with probability epsilon
            decision_probs[(node_i, node_j)] = min(epsilon, (1-epsilon))
            decision_probs[(node_j, node_i)] = max(epsilon, (1-epsilon))
        else:
            # give the correct decision
            decision_probs[(node_i, node_j)] = max(epsilon, (1-epsilon))
            decision_probs[(node_j, node_i)] = min(epsilon, (1-epsilon))


    return decision_probs