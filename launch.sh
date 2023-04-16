for dataset in child alarm asia cancer insurance; do
    for prior in 0 1; do
        for epsilon in 0.01 0.05 0.1; do
            python main.py --dataset $data --tabular 1 --epsilon $epsilon --algo global_scoring --uniform-prior $prior
        done
        python main.py --dataset $data --tabular 0 --algo global_scoring --uniform-prior $prior
        for tolerence in 0.1 0.25 0.5; do
            python main.py --dataset $data --tolerance $tolerance --tabular 0 --algo greedy --uniform-prior $prior
            for epsilon in 0.01 0.05 0.1; do
                python main.py --dataset $data --tolerance $tolerance --tabular 1 --epsilon $epsilon --algo greedy --uniform-prior $prior
            done
        done
    done
done