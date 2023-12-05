# Causal Discovery using Language Models
Stephanie Long, Alexandre Piché, Valentina Zantedeschi, Tibor Schuster, Alexandre Drouin (2023). [Structured Probabilistic Inference & Generative Modeling (SPIGM) workshop](https://spigmworkshop.github.io/). *International Conference in Machine Learning* (ICML). 

> Understanding the causal relationships that underlie a system is a fundamental prerequisite to accurate decision-making. In this work, we explore how expert knowledge can be used to improve the data-driven identification of causal graphs, beyond Markov equivalence classes. In doing so, we consider a setting where we can query an expert about the orientation of causal relationships between variables, but where the expert may provide erroneous information. We propose strategies for amending such expert knowledge based on consistency properties, e.g., acyclicity and conditional independencies in the equivalence class. We then report a case study, on real data, where a large language model is used as an imperfect expert.

[[Paper]](https://arxiv.org/abs/2307.02390)


![greedy_conf_main](https://github.com/StephLong614/Causal-disco-LLM-imperfect-experts/assets/17014892/3f13bfb8-e125-4c4b-887b-3c576b1a4e01)

# Running experiments

To run our greedy algorithm with the S_risk strategy (selecting at each iteration the edge that leads to the lowest risk of excluding the true graph)
```python3
python3 main.py --llm-engine text-davinci-002 --algo greedy_conf --dataset child --tol 0.1 --seed=965079
```

To run our greedy algorithm with the S_size strategy (selecting at each iteration the edge that leads to the smallest equivalence class)
```python3
python3 main.py --llm-engine text-davinci-002 --algo greedy_mec --dataset child --tol 0.1 --seed=965079
```

You can use text-davinci-003 instead, by specifying "--llm-engine text-davinci-002", or for calling the epsilon-expert, specify: "--tabular --epsilon <epsilon>"
Use "--tol" to select a different tolerance level.

[OpenAI text-davinci models will be deprecated on Jan 4th 2024](https://platform.openai.com/docs/models/gpt-3-5).
To reproduce our experiments, we cached the calls to OpenAI API in [text-davinci-002_llm_cache.pickle](https://github.com/StephLong614/Causal-disco-LLM-imperfect-experts/blob/main/text-davinci-002_llm_cache.pickle) and [text-davinci-003_llm_cache.pickle](https://github.com/StephLong614/Causal-disco-LLM-imperfect-experts/blob/main/text-davinci-003_llm_cache.pickle) for the seeds: 965079 79707 239916 537973 953100.
Our code automatically loads these pickles.

# Citing this work
Please use the following Bibtex entry to cite this work:
```
@misc{long2023causal,
      title={Causal Discovery with Language Models as Imperfect Experts}, 
      author={Stephanie Long and Alexandre Piché and Valentina Zantedeschi and Tibor Schuster and Alexandre Drouin},
      year={2023},
      eprint={2307.02390},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
