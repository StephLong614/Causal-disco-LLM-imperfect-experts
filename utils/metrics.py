from cdt.metrics import SHD
import networkx as nx
import numpy as np
from utils.dag_utils import list_of_tuples_to_digraph, order_graph 
from utils.plotting import plot_heatmap

def get_mec_shd(true_G, mec, args):
    """
    the graphs need to be ordered to be comparable
    """
    true_G = order_graph(true_G)
    target = nx.to_numpy_array(true_G)
    pred = np.stack([nx.to_numpy_array(list_of_tuples_to_digraph(dag)) for dag in mec], -1).sum(-1)
    pred = np.clip(pred, 0, 1)
    shd = SHD(target, pred, double_for_anticausal=False)
    return shd

    #for i, G in enumerate(mec):
    #    G = list_of_tuples_to_digraph(G)
    #    pred = nx.to_numpy_array(G)
    #    plot_heatmap(target-pred, lbls=G.nodes(), dataset=args.dataset, method=args.algo, name=f"graph_{i}.pdf")
    #    shd = SHD(target, pred, double_for_anticausal=False)
    #    shds[str(G)] = shd
    
    #return shds