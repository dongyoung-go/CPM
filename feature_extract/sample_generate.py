import argparse
import math
import random
import json
from datetime import date

import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, Dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-size', type=str, default='large')
    ap.add_argument('--dataset-type', default='Anthropic/hh-rlhf', choices=['Anthropic/hh-rlhf', 'stanfordnlp/shp'])
    ap.add_argument('--model-type', default='huggingface', choices=['openai', 'huggingface'])
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoding_params = {"temperature": 0.9,
    "top_p": 0.8,
    "top_k": 50,
    'max_new_tokens':150,                   
    }

    ## Load dataset
    dataset = load_dataset(args.dataset_type)
    train_data = dataset['train']
    test_data = dataset['test']
    test_data = test_data.filter(is_single_round).filter(lambda x: len(x["chosen"]) < 1000 and len(x["rejected"]) < 1000)

    ## Load proposal model

    out_fn = f'hh-generated_flan_t5_{args.model_size}.json'
    model_name = f"google/flan-t5-{args.model_size}"
    proposal_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    proposal_model.to(device)
    proposal_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ## Generate responses

    random.seed(42)
    samples_n = 100
    generate_n = 256
    f_bs = 16
    repeat_n = generate_n//f_bs
    bot_prefix = '\nAssistant: '
    idxs = [idx for idx in range(len(test_data))]
    random.shuffle(idxs)
    sampled_idxs = idxs[:samples_n]

    generated_results = dict()
    for idx in tqdm(sampled_idxs):
        prepend = bot_prefix.join(test_data[int(idx)]['chosen'].split(bot_prefix)[:-1]) + bot_prefix
        generated_result = []
        for _ in range(repeat_n):
            torch.cuda.empty_cache()
            inputs = tokenizer(prepend, return_tensors="pt")['input_ids']
            outputs = proposal_model.generate(inputs.to(device),num_return_sequences=f_bs,do_sample=True,**decoding_params)
            decode_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_result.append(decode_outputs)
            
        generated_result = sum(generated_result,[])
        generated_results[str(idx)] = {'prompt':prepend,'response':generated_result}
        with open(out_fn, 'w') as f:
            json.dump(generated_results, f)    

    df = pd.read_json(out_fn)
    dataset = Dataset.from_pandas(df.T)
    dataset.push_to_hub(f'dongyoung4091/{Path(out_fn).stem}')

def is_single_round(element):
    return (element["chosen"].count("\n\nHuman:") == 1 and
            element["chosen"].count("\n\nAssistant:") == 1 and
            element["rejected"].count("\n\nHuman:") == 1 and
            element["rejected"].count("\n\nAssistant:") == 1)

if __name__ == '__main__':
    main()
