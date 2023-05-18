import numpy as np
import openai
import pickle
import scipy

from utils.apis import pubmed_api


def get_lms_probs(undirected_edges, codebook):
  """
  return: dictionary of tuple and their likelihood of being wrong by the LM
  example {('Age', 'Disease'): 0.05, ...}
  """

  gpt3_decision_probs = {}

  for edge in undirected_edges:
      node_i = edge[0]
      node_j = edge[1]
      long_name_node_i = codebook.loc[codebook['var_name']==node_i, 'var_description'].to_string(index=False)
      long_name_node_j = codebook.loc[codebook['var_name']==node_j, 'var_description'].to_string(index=False)
      
      options = f"""
                Options:
                (A) According to medical doctors, {long_name_node_i} increases the risk of {long_name_node_j}
                (B) According to medical doctors, {long_name_node_j} increases the risk of {long_name_node_i}
                The answer is: 
                """
      
      log_scores = gpt3_scoring(options, options=['(A', '(B'], lock_token=' (')
      scores = scipy.special.softmax(-log_scores)
      
      if scores[0] > scores[1]:
        gpt3_decision_probs[(node_i, node_j)] = scores[0]
      else:
        gpt3_decision_probs[(node_j, node_i)] = scores[1]

  return gpt3_decision_probs


def calibrate(directed_edges, codebook):
  correct_answer = 0
  denom = 0

  for edge in directed_edges:
      # node_i -> node_j 
      node_i = edge[0]
      node_j = edge[1]
      long_name_node_i = codebook.loc[codebook['var_name']==node_i, 'var_description'].to_string(index=False)
      long_name_node_j = codebook.loc[codebook['var_name']==node_j, 'var_description'].to_string(index=False)
      options = f"""
                Options:
                (A) According to medical doctors, {long_name_node_i} increases the risk of {long_name_node_j}
                (B) According to medical doctors, {long_name_node_j} increases the risk of {long_name_node_i}
                The answer is: 
                """
      
      log_scores = gpt3_scoring(options, options=['(A)', '(B)'], lock_token=' (')
      scores = scipy.special.softmax(log_scores)
      if scores[0] > scores[1]:
        correct_answer += 1
      else:
        correct_answer += 0
      
      denom += 1

  estimated_error = 1 - (correct_answer/denom)
  return estimated_error


def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0, 
              logprobs=1, echo=False):
  LLM_CACHE = {}
  try:
      with open('llm_cache.pickle', 'rb') as f:
          LLM_CACHE = pickle.load(f)
  except:
      pass
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    response = LLM_CACHE[id]
  else:
    print('no cache hit, api call')
    response = openai.Completion.create(engine=engine, 
                                        prompt=prompt, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)
    LLM_CACHE[id] = response
    with open('llm_cache.pickle', 'wb') as f:
      pickle.dump(LLM_CACHE, f)
  return response


def gpt3_scoring(prompt, options, engine="text-davinci-003", verbose=False, n_tokens_score=9999999999, lock_token=None): 
    verbose and print("Scoring", len(options), "options") 
    gpt3_prompt_options = [f"{prompt}{o}" for o in options] 
    response = gpt3_call(engine=engine, prompt=gpt3_prompt_options, max_tokens=0, logprobs=1, temperature=0, echo=True,) 
    scores = [] 
    for option, choice in zip(options, response["choices"]): 
        if lock_token is not None: 
            n_tokens_score = choice["logprobs"]["tokens"][::-1].index(lock_token)
        tokens = choice["logprobs"]["tokens"][-n_tokens_score:] 
        verbose and print("Tokens:", tokens) 
        token_logprobs = choice["logprobs"]["token_logprobs"][-n_tokens_score:] 
        total_logprob = 0 
        denom = 0 
        for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)): 
            if token_logprob is not None: 
                denom += 1 
                total_logprob += token_logprob 
        scores.append(total_logprob) 
    return np.array(scores)


if __name__ == '__main__':
  options = """
            Options:
            (A) Cancer causes smoking
            (B) Smoking causes cancer
            The answer is: 
            """
  log_scores = gpt3_scoring(options, options=['(A)', '(B)'], lock_token=' (')
  scores = scipy.special.softmax(log_scores)
  print(scores)