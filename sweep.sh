#!/bin/bash
source /home/toolkit/.bashrc
conda activate noisy
cd /home/toolkit/git/Causal-disco/
for s in 965079 79707 239916 537973 953100
do
    for p in mec independent
    do
        for e in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
        do
            for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
            do
                for a in "greedy_mec" "greedy_conf"
                do
                    python3 main.py --seed=$s --tol $t --prior $p --algo $a --dataset $1 --wandb --tabular --epsilon $e
                done
            done

            for a in "greedy_bic" "global_scoring" "blind"
            do
                python3 main.py --seed=$s --tol $t --prior $p --algo $a --dataset $1 --wandb --tabular --epsilon $e
            done
        done
    done
done
python3 main.py --algo "PC" --dataset $1 --wandb