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