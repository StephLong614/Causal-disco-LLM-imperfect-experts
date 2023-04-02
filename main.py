import argparse
import numpy as np
import os
import pandas as pd

from utils.bayes import get_prior
from utils.data_generation import generate_dataset
from utils.plotting import plot_heatmap
from utils.dag_utils import get_undirected_edges, get_mec
from utils.metrics import get_mec_shd
from utils.language_models import get_lms_decisions


parser = argparse.ArgumentParser(description='Description of your program.')

# Add arguments
parser.add_argument('--algo', default="greedy", type=str, help='What algorithm to use')
parser.add_argument('--dataset', default="child", type=str, help='What dataset to use')
parser.add_argument('--pubmed-sources', type=int, help='How many PubMed sources to retrieve')
parser.add_argument('--verbose', default=0, type=int, help='Print')
parser.add_argument('--tolerance', default=0.101, type=float, help='algorithm error tolerance')

if __name__ == '__main__':
    args = parser.parse_args()

    match args.algo:
        case "greedy":
            from algo.greedy_search import greedy_search
            algo = greedy_search
        case "PC":
            algo = lambda gpt3_decision_probs, mec, undirected_edges, tol: (mec, {})
        case "global_scoring":
            from algo.global_scoring import global_scoring
            algo = global_scoring
    
    if not os.path.exists("_raw_bayesian_nets"):
        from utils.download_datasets import download_datasets
        download_datasets()
    
    try:
        codebook = pd.read_csv('codebooks/' + args.dataset + '.csv')
    except:
        print('cannot load the codebook')
        codebook = None

    true_G, data = generate_dataset('_raw_bayesian_nets/' + args.dataset + '.bif', n=1000, seed=0)
    plot_heatmap(true_G, 'figures/true_g.pdf')
    undirected_edges = get_undirected_edges(true_G, verbose=args.verbose)
    lms_decisions = get_lms_decisions(undirected_edges, codebook)
    mec = get_mec(true_G)
    prior_prob = get_prior(undirected_edges, mec)
    breakpoint()
    new_mec, decisions = algo(lms_decisions, mec, undirected_edges, tol=args.tolerance)
    shds = get_mec_shd(true_G, new_mec)
    shds_scores = np.array([v for v in shds.values()])
    print('Average SHD for the mec: ', np.mean(shds_scores))
    #for k, v in shds.items():
    #    print(v)
    #print('PC mec size: ', len(mec))
    #print('Greedy mec size: ', len(new_mec))