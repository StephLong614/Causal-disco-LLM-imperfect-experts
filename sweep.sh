w="noisy-expert-rnd"
for e in "text-davinci-002" "text-davinci-003"
    for d in child alarm asia insurance
    do  
        for p in mec independent
        do
            for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.
            do
                for a in "greedy_mec" "greedy_conf"
                do
                    python3 main.py --llm-engine $e --tol $t --prior $p --algo $a --dataset $d --wandb --wandb-project $w
                done
            done

            for a in "blind"
            do
                python3 main.py --llm-engine $e --prior $p --algo $a --dataset $d --wandb --wandb-project $w
            done
        done
    done
done