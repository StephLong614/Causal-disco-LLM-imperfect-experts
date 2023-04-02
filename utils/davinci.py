import numpy as np
import openai
import pickle

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


#engine="gpt-3.5-turbo"
#engine="text-davinci-003"
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
