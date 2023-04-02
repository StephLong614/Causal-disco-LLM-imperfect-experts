import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


def plot_heatmap(G, name=''):
    lbls = G.nodes()
    g = nx.to_numpy_array(G)
    ax = sns.heatmap(g)
    ax.set_xticklabels(lbls, rotation=90)
    ax.set_yticklabels(lbls, rotation=0)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

