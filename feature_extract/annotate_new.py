# This is a fork of shp4.py which focuses only on annotating samples with LM features
from tqdm import tqdm
import dill as pickle
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, HuggingFacePipeline
from langchain.chains import LLMChain
import numpy as np
from pathlib import Path
import argparse
import re
import sys
sys.path.append('../')
from utils.constant import FEATURES
from utils.common import get_lm, feature_score,get_feature_extractor
from functools import partial
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', default='google/flan-t5-xl')
    ap.add_argument('--model-type', default='huggingface', choices=['openai', 'huggingface'])

    ap.add_argument('--dataset_name', default='dongyoung4091/hh-rlhf_with_features_flan_t5_large')
    ap.add_argument('--options', nargs='+', default=['chosen', 'rejected'])
    ap.add_argument('--context', default='human')
    ap.add_argument('--response-prefix', default='assistant')
    ap.add_argument('--split', choices = ['train','test'])
    ap.add_argument('--eval-generated',action='store_true',default=False)
    args = ap.parse_args()

    if args.eval_generated:
        def all_features_scores(row, feature_extractor, features):
            for feature, feature_kwargs in features.items():
                prompt = row['prompt']
                for prefix in ['\n\nHuman: ','n\nAssistant: ','\n\nPOST: ','\n\nResponse: ']:
                    prompt = prompt.replace(prefix,'')
                if type(row['response'])==list:
                    score_list = []
                    for response in row['response']:
                        score_list.append(feature_score(feature_extractor, history=prompt ,reply=response, **feature_kwargs))
                    row[f'zeroshot_{feature}'] = score_list
                else:
                    row[f'zeroshot_{feature}'] = feature_score(feature_extractor, history=prompt ,reply=row['response'], **feature_kwargs)
            return row
    else:
        def all_features_scores(row, feature_extractor, features):
            for feature, feature_kwargs in features.items():
                for option in args.options:
                    row[f'zeroshot_{feature}_{option}'] = feature_score(feature_extractor, history=row[args.context] ,reply=row[f"{args.response_prefix}_{option}"], **feature_kwargs)
            return row
    
    # Load feature extractor
    lm = get_lm(args.model_name, args.model_type)
    feature_extractor = get_feature_extractor(lm, data_type='hh' if 'hh' in args.dataset_name else 'shp')

    # Load dataset
    ds = load_dataset(args.dataset_name, split=args.split)
    dataset_name_short = args.dataset_name.split('/')[-1]

    # Annotate
    annotate = partial(all_features_scores, feature_extractor=feature_extractor, features=FEATURES)
    ds = ds.map(annotate)

    # Save
    save_fname = f'out/{dataset_name_short}_{args.split}'
    if 'xl' not in args.model_name:
        save_fname += '_' + args.model_name.split('/')[-1]
    ds.save_to_disk(save_fname)
    print(f"Saved to {save_fname}")
    
main()