import argparse
from causaldag import DAG
import networkx as nx
import os
import pandas as pd
import wandb
import numpy as np
import random

from models.noisy_expert import NoisyExpert
from models.oracles import EpsilonOracle

from utils.data_generation import generate_dataset
from utils.plotting import plot_heatmap
from utils.dag_utils import get_undirected_edges, get_mec
from utils.metrics import get_mec_shd
from utils.language_models import get_lms_probs, calibrate

random.seed(4)
np.random.seed(4)

parser = argparse.ArgumentParser(description='Description of your program.')

# Add arguments
parser.add_argument('--algo', default="greedy", type=str, help='What algorithm to use')
parser.add_argument('--prior', default="mec", choices=["mec", "independent"])
parser.add_argument('--probability', default="posterior", choices=["posterior", "prior", "likelihood"])
parser.add_argument('--dataset', default="child", type=str, help='What dataset to use')
parser.add_argument('--pubmed-sources', type=int, help='How many PubMed sources to retrieve')
parser.add_argument('--verbose', default=0, type=int, help='For debugging purposes')
parser.add_argument('--tabular', default=0, type=int, help='If 1 use tabular expert, else use gpt3')
parser.add_argument('--uniform-prior', default=0, type=int, help='If set to 1 we will use the uniform prior over edges')
parser.add_argument('--epsilon', default=0.05, type=float, help='algorithm error tolerance')
parser.add_argument('-tol', '--tolerance', default=0.101, type=float, help='algorithm error tolerance')
parser.add_argument('--wandb', default=False, action="store_true", help='to log on wandb')

if __name__ == '__main__':
    args = parser.parse_args()
    wandb.login(key='246c8f672f0416b12172d64574c12d8a7ddae387')

    wandb.init(config=args,
               project='causal discovery with LMs',
               mode=None if args.wandb else 'disabled'
               )

    match args.algo:
        case "greedy":
            from algo.greedy_search import greedy_search
            algo = greedy_search
        case "greedy2":
            from algo.greedy_search2 import greedy_search
            algo = greedy_search
        case "PC":
            algo = lambda gpt3_decision_probs, prior_prob, mec, undirected_edges, tol: (mec, {})
        case "global_scoring":
            from algo.global_scoring import global_scoring
            algo = global_scoring
    
    match args.prior:
        case "mec":
            from models.priors import MECPrior
            prior = MECPrior
        
        case "independent":
            from models.priors import IndependentPrior
            prior = IndependentPrior
    
    if not os.path.exists("_raw_bayesian_nets"):
        from utils.download_datasets import download_datasets
        download_datasets()

    true_G, data = generate_dataset('_raw_bayesian_nets/' + args.dataset + '.bif', n=1000, seed=0)
    cpdag, mec = get_mec(true_G)

    plot_heatmap(nx.to_numpy_array(true_G), lbls=true_G.nodes(), dataset=args.dataset, name='true_g.pdf')
    undirected_edges = get_undirected_edges(true_G, verbose=args.verbose)

    #lm_error = calibrate(directed_edges, codebook) 
    #print('Calibration Error: ', lm_error)

    if args.tabular:
        oracle = EpsilonOracle(undirected_edges, epsilon=args.epsilon)
        observations = oracle.decide_all()

    else:
        try:
            codebook = pd.read_csv('codebooks/' + args.dataset + '.csv')
        except:
            print('cannot load the codebook')
            codebook = None
    
        observations = get_lms_probs(undirected_edges, codebook)

    print("\nTrue Orientations:", undirected_edges)
    print("\nOrientations given by the expert:", observations)
    prior = MECPrior(cpdag)
    model = NoisyExpert(prior, oracle.likelihoods)

    match args.probability:
        case "posterior":
            prob_method = model.posterior
        
        case "likelihood":
            prob_method = model.likelihood

        case "prior":
            prob_method = lambda _, edges: prior(edges)

    new_mec, decisions, p_correct = algo(observations, prob_method, mec, undirected_edges, tol=args.tolerance)
    shd = get_mec_shd(true_G, new_mec, args)
    #shds_scores = np.array([v for v in shds.values()])
    print('Probability true DAG is in final MEC: %.3f' % p_correct)
    print("Final MEC' SHD: ", shd)
    print('MEC size: ', len(new_mec))
    wandb.log({'mec size': len(new_mec),
               'shd': shd})
    wandb.finish()
    #for k, v in shds.items():
    #    print(v)
    #print('PC mec size: ', len(mec))
    #print('Greedy mec size: ', len(new_mec))