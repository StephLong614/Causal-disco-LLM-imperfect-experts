import numpy as np
import openai
import pickle
import scipy

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


def gpt3_scoring(options, engine="text-davinci-003", verbose=False, print_tokens=False):
  verbose and print("Scoring", len(options), "options")
  gpt3_prompt_options = options
  response = gpt3_call(
      engine=engine, 
      prompt=gpt3_prompt_options, 
      max_tokens=0,
      logprobs=1, 
      temperature=0,
      echo=True,)
  scores = []
  for option, choice in zip(options, response["choices"]):
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]
    total_logprob = 0
    denom = 0
    for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
      if token_logprob is not None:
        denom += 1
        total_logprob += token_logprob

    scores.append(total_logprob / denom)
  
  return np.array(scores)


def get_lms_decisions(undirected_edges, codebook, gpt3_error=0.05):
  """
  return: dictionary of tuple and their likelihood of being wrong by the LM
  example {('Age', 'Disease'): 0.05, ...}
  """

  gpt3_decisions = []
  gpt3_decision_probs = {}

  for edge in undirected_edges:
      node_i = edge[0]
      node_j = edge[1]
      long_name_node_i = codebook.loc[codebook['var_name']==node_i, 'var_description'].to_string(index=False)
      #print(long_name_node_i)
      #print(node_j)
      long_name_node_j = codebook.loc[codebook['var_name']==node_j, 'var_description'].to_string(index=False)
      options = [f'According to medical doctors, {long_name_node_i} increases the risk of {long_name_node_j}',
                f'According to medical doctors, {long_name_node_j} increases the risk of {long_name_node_i}']
      
      log_scores = gpt3_scoring(options)
      scores = scipy.special.softmax(log_scores)
      # if scores[0] is greater than scores[1], gpt3 believes node_i -> node_j
      if (scores[0] > scores[1]):
          decision = (node_j, node_i)
          gpt3_decision_probs[(node_i, node_j)] = gpt3_error
          gpt3_decision_probs[(node_j, node_i)] = 1 - gpt3_error
      else:
          decision = (node_j, node_i)
          gpt3_decision_probs[(node_j, node_i)] = gpt3_error
          gpt3_decision_probs[(node_i, node_j)] = 1 - gpt3_error
      
      gpt3_decisions.append(decision)

  return gpt3_decision_probs

if __name__ == '__main__':
    options = ['Smoking causes cancer', 'Cancer causes smoking']

    log_scores = gpt3_scoring(options)
    scores = scipy.special.softmax(log_scores)
    print(scores)