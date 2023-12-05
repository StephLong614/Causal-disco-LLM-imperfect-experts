#!/bin/bash
d=$1 # dataset
w=$2 # wandb project
for s in 965079 79707 239916 537973 953100
do
    for e in "text-davinci-002" "text-davinci-003"
    do
        for p in mec independent
        do
            for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
            do
                for a in "greedy_mec" "greedy_conf"
                do
                    python3 main.py --seed=$s --tol $t --prior $p --algo $a --dataset $d --wandb --llm-engine $e --wandb-project $w
                done
            done

            python3 main.py --seed=$s --prior $p --algo "naive" --dataset $d --wandb --llm-engine $e --wandb-project $w

        done
    done
done
python3 main.py --algo "PC" --dataset $d --wandb --wandb-project $w
