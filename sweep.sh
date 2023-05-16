for d in child alarm asia cancer insurance
do    
    for p in mec independent
    do
        for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
        do
            for a in "greedy_mec" "greedy_conf"
            do
                python3 main.py --tol $t --prior $p --algo $a --dataset $d --wandb
            done
        done

        for a in "greedy_bic" "global_scoring"
        do
            python3 main.py --tol $t --prior $p --algo $a --dataset $d --wandb
        done
    done
    python3 main.py --tol $t --prior $p --algo "PC" --dataset $d --wandb
done