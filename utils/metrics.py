from cdt.metrics import SHD
import networkx as nx
from utils.dag_utils import list_of_tuples_to_digraph, order_graph 

def get_mec_shd(true_G, mec, args):
    """
    the graphs need to be ordered to be comparable
    """
    shds = {}
    true_G = order_graph(true_G)
    target = nx.to_numpy_array(true_G)

    for i, G in enumerate(mec):
        pred = list_of_tuples_to_digraph(G)
        #plot_graph(pred, f"figures/graph_{i}.pdf")
        pred = nx.to_numpy_array(pred)
        shd = SHD(target, pred, double_for_anticausal=False)
        #breakpoint()
        shds[str(G)] = shd
    
    return shds