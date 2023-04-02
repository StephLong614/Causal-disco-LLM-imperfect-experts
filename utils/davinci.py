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


def gpt3_scoring(prompt, options, engine="text-davinci-003", verbose=False, n_tokens_score=9999999999, lock_token=None): 
    verbose and print("Scoring", len(options), "options") 
    gpt3_prompt_options = [f"{prompt}{o}" for o in options] 
    response = gpt3_call(engine=engine, prompt=gpt3_prompt_options, max_tokens=0, logprobs=1, temperature=0, echo=True,) 
    scores = [] 
    for option, choice in zip(options, response["choices"]): 
        if lock_token is not None: 
            n_tokens_score = choice["logprobs"]["tokens"][::-1].index(lock_token) + 1
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