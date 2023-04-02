import openai
import pickle
import pymed
from pymed import PubMed

def get_abstracts(query, n=1):
    # Create a PubMed object
    #pubmed = PubMed(tool="Causal Discovery ChatGPT", email="sclong614@gmail.com")
    pubmed = PubMed(tool="MyTool", email="my@email.address")
    
    # Execute the query
    results = pubmed.query(query, max_results=n)
    
    # Extract the abstracts
    abstracts = []
    for article in results:
        abstract = article.abstract
        abstracts.append(abstract)
    
    breakpoint()
    return abstracts

def chatgpt_scoring(options, verbose=False, print_tokens=False):
  verbose and print("Scoring", len(options), "options")
  gpt3_prompt_options = options
  response = chatgpt_call(
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


def chatgpt_call(prompt="", max_tokens=128, temperature=0, 
                 logprobs=1, echo=False):
    engine="gpt-3.5-turbo"
    CHATGPT_LLM_CACHE = {}
    try:
        with open('chat_gpt_llm_cache.pickle', 'rb') as f:
            CHATGPT_LLM_CACHE = pickle.load(f)
    except:
        pass
    full_query = ""
    for p in prompt:
        full_query += p
    id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
    if id in CHATGPT_LLM_CACHE.keys():
        response = CHATGPT_LLM_CACHE[id]
        return response 

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system",
                 "content": "You are a helpful medical assistant. Answer with a short query for the pubmed search engine."},
                {"role": "user", 
                 "content": prompt[0]},
            ],
        temperature=0.,
        n=1
    )

    answer = [choice.message.content for choice in completion.choices]
    print(answer)
    abstracts = get_abstracts(query=answer[0])
    abstract = "Abstract: " + abstracts[0]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Only answer with single word the user's question based on the following medical abstact."},
                {"role": "user", "content": abstract + prompt[0]},
            ],
        max_tokens=max_tokens, 
        temperature=temperature,
        logprobs=logprobs,
        echo=echo
    )

    CHATGPT_LLM_CACHE[id] = response
    with open('chat_gpt_llm_cache.pickle', 'wb') as f:
      pickle.dump(CHATGPT_LLM_CACHE, f)
    breakpoint()
    return response

if __name__ == '__main__':
    user_utterance = "USER: Does age increase the risk of cancer?"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Answer with a short query for the pubmed search engine."},
                {"role": "user", "content": user_utterance},
            ],
        temperature=0.01,
        n=1
    )

    answer = [choice.message.content for choice in completion.choices]
    print(answer)
    abstracts = get_abstracts(query=answer[0])
    print(abstracts)

    abstract = "Abstract: " + abstracts[0]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Only answer with single word the user's question based on the following medical abstact."},
                {"role": "user", "content": abstract + user_utterance},
            ],
        temperature=0.01,
        n=1
    )

    answer = [choice.message.content for choice in completion.choices]
    print(answer)
