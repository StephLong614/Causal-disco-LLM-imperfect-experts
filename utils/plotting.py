import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import seaborn as sns


def plot_heatmap(g, lbls, dataset='', method=None, name='', base_dir='figures/'):
    dir_ = base_dir + dataset + '/' 
    if method:
        dir_ = dir_ + method + '/'
    
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    ax = sns.heatmap(g)
    ax.set_xticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=90)
    ax.set_yticks(range(len(lbls)))
    ax.set_yticklabels(lbls, rotation=0)
    plt.tight_layout()
    plt.savefig(dir_ + name)
    plt.close()

