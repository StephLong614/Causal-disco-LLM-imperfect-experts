from causaldag import DAG
import networkx as nx
import numpy as np

def get_mec(G_cpdag):

    mec = []

    for dag in G_cpdag.all_dags():
        g = [edge for edge in dag]
        mec.append(g)
    
    return mec

def get_undirected_edges(true_G, verbose=False):

    dag = DAG.from_nx(true_G)
    edges = dag.arcs - dag.cpdag().arcs

    if verbose:
        print("Unoriented edges: ", edges)
    
    return edges

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

def order_graph(graph):
    H = nx.DiGraph()
    #print(graph.nodes)
    H.add_nodes_from(sorted(graph.nodes(data=True)))
    H.add_edges_from(graph.edges(data=True))
    return H


def list_of_tuples_to_digraph(list_of_tuples):
    G = nx.DiGraph()
    # Add nodes best_graph
    for edge in list_of_tuples:
        node_i = edge[0]
        node_j = edge[1]
        G.add_edge(node_i, node_j)
    G = order_graph(G)
    return G

def is_dag_in_mec(G, mec):

    for dag in mec:
        ans = True
        for edge in dag:
            if edge not in G.edges:
                ans = False
                break
        if ans:
            return 1.
        
    return 0.