import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from datasets import load_dataset
import argparse
import os

def find_score(score_list):
    result_list = []
    for gen_sentence in score_list:
        search1 = re.findall(r'^\s*([\d.]+)\s+([\d.]+)\s*$', gen_sentence, re.MULTILINE)
        search2 = re.findall(r'(?:Assistant 1:|Assistant 2:)\s+(\d+)', gen_sentence)
        if len(search1)>0:
            result_list.append(' '.join(search1[0]))
        elif len(search2)>0:
            result_list.append(' '.join(search2))
        else:
            result_list.append('0 0')
    return result_list

# Define constant
URL = 'https://api.anthropic.com/v1/complete'
body = {
  "prompt": None,
  "model": "claude-2",
  "max_tokens_to_sample": 13,
  "temperature": 0,
  "top_p": 1,
}

headers = {
    "accept": "application/json",
    "anthropic-version": f"2023-06-01",
    "content-type":"application/json",
    "x-api-key": 'API_KEY'
}

Human_prefix = '\n\nHuman: '
Assistant_prefix = '\n\nAssistant: '
# prompt with accuracy 0.577
hh_prompt_format = """

Human:
You are a helpful and precise assistant for checking the quality of the answer. We would like to request your feedback on the performance of two AI assistants in response to the user question displayed below.

[Question]
{query}
[The Start of Assistant 1’s Answer]
{answer1}
[The Start of Assistant 2’s Answer]
{answer2}

Please rate the helpfulness, relevance, accuracy, level of details of their responses.
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

Assistant:"""
# prompt with accuracy 0.612 (hh_prompt_format scored 0.581 in shp dataset)
shp_prompt_format = """

Human:
[System]
You are a helpful and precise assistant for checking the quality of the answer. We would like to request your feedback on the performance of two AI assistants in response to the user question displayed below.

[Question]
{query}

[The Start of Assistant 1’s Answer]
{answer1}

[The Start of Assistant 2’s Answer]
{answer2}

[System]
Please rate the helpfulness, relevance, accuracy, level of details of their responses.
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

Assistant:"""

def main():
    ap = argparse.ArgumentParser(description=
            """
            Evaluates the pairwise ranking accuracy of generated responses.
            """)
    ap.add_argument('--dataset', default='dongyoung4091/hh-generated_flan_t5_large')
    ap.add_argument('--split', default='train')
    ap.add_argument('--indices', nargs='+', type=int, default=[0, 1])
    args = ap.parse_args()
    
    assert len(args.indices)==2,' It should be pairwise comparision'
    identifier = '_'.join([str(idx) for idx in args.indices])
    # Load dataset
    if 'feather' in args.dataset:
        dataset = pd.read_feather(args.dataset)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    dataset_name = args.dataset.split('/')[-1]
    save_fname = f'./data/{dataset_name}_claude_evaluation{identifier}.json'
    if os.path.exists(save_fname):
        raise FileExistsError(f"'{save_fname}' already exists.")    
    if 'hh-generated' in args.dataset:
        prompt_format = hh_prompt_format
    elif 'shp-generated' in args.dataset:
        prompt_format = shp_prompt_format

    # Evaluate
    prompt_list = []
    prompt_swap_list = []
    response_list = []
    response_swap_list = []
    original_prompt_list = []
    original_response1_list = []
    original_response2_list = []
    for idx in tqdm(range(len(dataset))):
        original_query = dataset['prompt'][idx]
        query=original_query.replace(Human_prefix,'').replace(Assistant_prefix,'')
        original_prompt_list.append(original_query)
        
        answer1=dataset['response'][idx][args.indices[0]]
        answer2=dataset['response'][idx][args.indices[1]]
        prompt = prompt_format.format(query=query,answer1=answer1,answer2=answer2)
        original_response1_list.append(answer1)
        original_response2_list.append(answer2)
        prompt_list.append(prompt)
        
        body.update({'prompt':prompt})
        response = requests.post(URL, headers=headers,json=body)
        response = response.json()['completion']
        response_list.append(response)
        
        answer1=dataset['response'][idx][args.indices[1]]
        answer2=dataset['response'][idx][args.indices[0]]
        prompt = prompt_format.format(query=query,answer1=answer1,answer2=answer2)
        prompt_swap_list.append(prompt)
        
        body.update({'prompt':prompt})
        response = requests.post(URL, headers=headers,json=body)
        response = response.json()['completion']
        response_swap_list.append(response)    

    # Post process
    final_list = {'prompt':prompt_list,'response':find_score(response_list),
                'prompt_swap':prompt_swap_list,'response_swap':find_score(response_swap_list)}

    scoreset1_list = []
    scoreset2_list = []
    for scoreset1,scoreset2 in zip(final_list['response'],final_list['response_swap']):
        scoreset1 = [float(x.strip()) for x in scoreset1.split(' ')]
        scoreset2 = [float(x.strip()) for x in scoreset2.split(' ')]
        scoreset1_list.append(scoreset1)
        scoreset2_list.append(scoreset2)

    assistant1 = np.vstack([np.array(scoreset1_list)[:,0],np.array(scoreset2_list)[:,1]]).mean(0).tolist()
    assistant2 = np.vstack([np.array(scoreset1_list)[:,1],np.array(scoreset2_list)[:,0]]).mean(0).tolist()

    final_list['assistant1']=assistant1
    final_list['assistant2']=assistant2
    final_list['original_prompt']=original_prompt_list
    final_list['original_response1']=original_response1_list
    final_list['original_response2']=original_response2_list

    # Save
    with open(save_fname, encoding="utf-8", mode="w") as file:
        json.dump(final_list, file)
    print(f"Saved to {save_fname}")

main()