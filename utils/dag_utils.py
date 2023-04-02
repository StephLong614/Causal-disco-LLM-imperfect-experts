from causaldag import DAG
import networkx as nx

def get_mec(true_G):
    nodes = list(true_G.nodes())
    G_cpdag = DAG.from_amat(nx.adjacency_matrix(true_G)).cpdag()
    mec = []

    for dag in G_cpdag.all_dags():
        g = []
        for edge in dag:
            i = edge[0]
            j = edge[1]
            g.append((nodes[i], nodes[j]))
        mec.append(g)
    return mec

def get_directed_edges(true_G, verbose=True):
    G_cpdag = DAG.from_amat(nx.adjacency_matrix(true_G)).cpdag()
    # List all undirected edges
    nodes = list(true_G.nodes())
    A_cpdag = G_cpdag.to_amat()[0]
    directed_edges = []

    edges_count = {}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not (A_cpdag[i, j] == 1 and A_cpdag[j, i] == 1):
                if verbose:
                    print(f"Undirected: {nodes[i]} <-> {nodes[j]}")
                edges_count[f"{nodes[i]} <-> {nodes[j]}"] = [0,0]
                #edges_count[f"{nodes[j]} -> {nodes[i]}"] = 0
                directed_edges.append((nodes[i], nodes[j]))
                
    return directed_edges

def get_undirected_edges(true_G, verbose=True):
    G_cpdag = DAG.from_amat(nx.adjacency_matrix(true_G)).cpdag()
    # List all undirected edges
    nodes = list(true_G.nodes())
    A_cpdag = G_cpdag.to_amat()[0]
    undirected_edges = []

    edges_count = {}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if A_cpdag[i, j] == 1 and A_cpdag[j, i] == 1:
                if verbose:
                    print(f"Undirected: {nodes[i]} <-> {nodes[j]}")
                edges_count[f"{nodes[i]} <-> {nodes[j]}"] = [0,0]
                #edges_count[f"{nodes[j]} -> {nodes[i]}"] = 0
                undirected_edges.append((nodes[i], nodes[j]))
                
    return undirected_edges

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