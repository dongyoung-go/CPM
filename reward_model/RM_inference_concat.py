import time
import sys
# This is a fork of shp4.py which focuses only on annotating samples with LM features
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re
import sys
sys.path.append('../')
from utils.constant import FEATURES
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', default='google/flan-t5-xl')
    ap.add_argument('--samples-name', 
                    # default='dongyoung4091/shp-generated_flan_t5_large',
                    default='dongyoung4091/hh-generated_flan_t5_large'
                )
    ap.add_argument('--model-path', 
                    # default='./model_ckpt/flan-t5-xl_dongyoung4091_shp_with_features_20k_lr1e-05_shp-prepend_split_1234'
                    default='./model_ckpt/flan-t5-xl_dongyoung4091_hh-rlhf_with_features_lr1e-05_prepend_split_1234_subset_100'
                )
    ap.add_argument('--save_name', default='')
    args = ap.parse_args()

    # set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Load model
    model_name = args.model_name

    # Load the pretrained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    linear = torch.nn.Linear(model.config.hidden_size, 1)
    model.set_output_embeddings(linear)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    ckpt = torch.load(args.model_path, map_location='cpu')
    print(ckpt.keys())
    model.load_state_dict(ckpt['model'])
    # model.save_pretrained(fname)
    model.to(device)
    model.eval()

    from datasets import load_dataset
    dataset = load_dataset(args.samples_name)
    t5_samples = dataset['train'].to_pandas()
    t5_samples = t5_samples[['prompt','response']].T.to_dict()

    if args.save_name=='':
        args.save_name = args.samples_name.split('/')[-1] +'_with_' + args.model_path.split('/')[-1] +'.json'
    print(f"args.save_name: {args.save_name}")

    st_time = time.time()
    max_samples=100; max_responses=256
    annotated_samples = []

    for _, entry in tqdm(list(t5_samples.items())[:max_samples]):
        prompt = entry['prompt']
        if 'shp' in args.model_path:
            prompt=prompt.replace('\n\nHuman: ','\n\POST: ').replace('\n\nAssistant: ','\n\nResponse: ')
        elif 'hh' in args.model_path:
            prompt=prompt.replace('\n\nPOST: ','\n\nHuman: ').replace('\n\nResponse: ','\n\nAssistant: ')
        for response in tqdm(entry['response'][:max_responses], leave=False):
            ## Model Score part start
            with torch.no_grad():
                prompt_ids = tokenizer.encode(prompt, return_tensors='pt',padding="max_length", max_length = 512, truncation=True)
                response_ids = tokenizer.encode(response, return_tensors='pt',padding="max_length", max_length = 512, truncation=True)
                result = model(input_ids=prompt_ids.to(model.device), decoder_input_ids=response_ids.to(model.device)).logits[:,:,0]
                # get last_hidden_state w last token
                result_last = result[:,-1]
                score = result_last.item()
            ## Model Score part end
            if 'shp' in args.samples_name:
                prompt=prompt.replace('\n\nHuman: ','\n\nPOST: ').replace('\n\nAssistant: ','\n\nResponse: ')
            elif 'hh' in args.samples_name:
                prompt=prompt.replace('\n\nPOST: ','\n\nHuman: ').replace('\n\nResponse: ','\n\nAssistant: ')
            entry = {'prompt': prompt, 'response': response,'score': score}
            annotated_samples.append(entry)
    annotated_samples = pd.DataFrame(annotated_samples, columns=['response', 'prompt','score'])
    annotated_samples.to_json(args.save_name)
    ed_time = time.time()
    print(f"Saved to {args.save_name}. Took {(ed_time-st_time)/3600:0.3f}h")
main()