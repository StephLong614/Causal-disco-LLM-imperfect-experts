import numpy as np
import openai
import pickle
import scipy
from utils.davinci import gpt3_scoring
from utils.chatgpt import chatgpt_scoring



def get_lms_decisions(undirected_edges, codebook, gpt3_error=0.05, engine='chatgpt'):
  """
  return: dictionary of tuple and their likelihood of being wrong by the LM
  example {('Age', 'Disease'): 0.05, ...}
  """

  #TODO: flip error for likelihood

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
      
      if engine == 'davinci':
        log_scores = gpt3_scoring(options)
      else:
        log_scores = chatgpt_scoring(options)
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


def calibrate(directed_edges, codebook):
  correct_answer = 0
  denom = 0

  for edge in directed_edges:
      node_i = edge[0]
      node_j = edge[1]
      long_name_node_i = codebook.loc[codebook['var_name']==node_i, 'var_description'].to_string(index=False)
      long_name_node_j = codebook.loc[codebook['var_name']==node_j, 'var_description'].to_string(index=False)
      options = [f'According to medical doctors, {long_name_node_i} increases the risk of {long_name_node_j}',
                 f'According to medical doctors, {long_name_node_j} increases the risk of {long_name_node_i}']
     
      log_scores = gpt3_scoring(options)
      scores = scipy.special.softmax(log_scores)
      if scores[0] > scores[1]:
        correct_answer += 1
      else:
        correct_answer += 0
      
      denom += 1

  estimated_error = 1 - (correct_answer/denom)
  return estimated_error


if __name__ == '__main__':
    options = ['Smoking causes cancer', 'Cancer causes smoking']

    log_scores = gpt3_scoring(options)
    scores = scipy.special.softmax(log_scores)
    print(scores)