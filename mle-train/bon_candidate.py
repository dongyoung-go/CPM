import sys
sys.path.append('../feature_extract')
from reward_scaling_lll import *
from common import *

import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk
import pandas as pd
import dill as pickle

def standardize(data):
    data -= data.mean()
    data /= data.std()
    return data


def load_and_add_score(scored_samples, model_name):
    if 'pkl' in model_name:
        # LLL scores are saved in pkl format
        model_load = pickle.load(open(model_name, 'rb'))
        model_name_short = model_name.split('/')[-1]
        scored_samples = add_callback_features(scored_samples, model_load['feature_callbacks'])
        features_values_matrix = features_values(scored_samples, model_load['feature_names'])
        scored_samples = add_model_scores(model_name_short, scored_samples, model_load, features_values_matrix,standardize=True)
    elif 'json' in model_name:
        # PM scores are saved in json format
        pm_score = pd.read_json(model_name)
        pm_score = pm_score.drop_duplicates()
        model_name_short = 'pm_score'
        pm_score[model_name_short] = standardize(pm_score['score'].values.reshape(1,-1)).reshape(-1)
        scored_samples = scored_samples.merge(pm_score,on=['prompt','response'],how='inner')
    else:
        # external PM with datasets
        try:
            pm_score = load_dataset(model_name)['train'].to_pandas()
        except:
            pm_score = load_from_disk(model_name)['train'].to_pandas()
        pm_score = pm_score.drop_duplicates()
        model_name_short = 'external_rm1'
        pm_score[model_name_short] = pm_score['external_rm1']
        scored_samples = scored_samples.merge(pm_score[['response','prompt',model_name_short]],on=['prompt','response'],how='inner')
        print('scored_samples.columns',scored_samples.columns)
        scored_samples[model_name_short] = standardize(scored_samples[model_name_short].values.reshape(1,-1)).reshape(-1)

    return scored_samples, model_name_short

def make_comparison_data(scored_samples,args):
    scored_samples_name = str(args.scored_samples).split('/')[-1]
    scored_samples = scored_samples.dropna()  # some features might not have been properly scored
    if args.additional_samples:
        reward_ds = load_dataset(args.additional_samples)['train'].to_pandas()
        reward_ds = reward_ds.drop_duplicates(subset=['prompt','response']) # as some response are trivial, they are duplicated (and have same reward score)
        scored_samples = scored_samples.merge(reward_ds,on=['prompt','response'],how='left')

    model_names = []
    for fname in args.model_paths:
        scored_samples, col_name = load_and_add_score(scored_samples, fname)
        # preserve candidate names
        model_names.append(col_name)
        
    scored_samples['vanila_score'] = 1

    value_1 = 'vanila_score'
    for model_name in model_names:
        value_2 = model_name

        # Calculate argmax response (i.e. BoN)
        max_value_1_rows = scored_samples.groupby('prompt')[value_1].idxmax()
        max_value_2_rows = scored_samples.groupby('prompt')[value_2].idxmax()

        prompts = scored_samples.loc[max_value_2_rows]['prompt'].tolist()
        max_value_1_responses = scored_samples.loc[max_value_1_rows]['response'].tolist()
        max_value_2_responses = scored_samples.loc[max_value_2_rows]['response'].tolist()

        result_scored_samples = pd.DataFrame({'prompt': prompts,
                                'response': [[r1,r2] for r1,r2 in zip(max_value_1_responses,max_value_2_responses)],
                                })
        save_fname = f'../mle-train/out/{scored_samples_name}_{value_1}_vs_{value_2}.feather'
        result_scored_samples.reset_index(drop=True).to_feather(save_fname)
        print(result_scored_samples['response'].head())
        print(f"Saved to f{save_fname}")
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored-samples', type=Path, default='dongyoung4091/shp-generated_flan_t5_large_with_features')
    ap.add_argument('--additional-samples', help='to add features on scored_samples')
    ap.add_argument('--model-paths', nargs='+',  default= ["../mle-train/out/shp_with_features_20k_flan_t5_large_train_logistic.pkl",
                "../mle-train/out/shp_with_features_20k_flan_t5_large_train_coefficients.pkl",
                '../reward_model/shp-generated_flan_t5_large_annotated_google_flan-t5-xl_train.json',
                ])
    ap.add_argument('--chunk_eval', action='store_true', help='chunk_eval for calculating std')
    args = ap.parse_args()

    scored_samples = load_scored_samples(args.scored_samples)
    if args.chunk_eval:
        scored_samples_origin = scored_samples.copy()
        name_origin = str(args.scored_samples)
        n_of_bon = 16
        for chunk in range(256//n_of_bon):
            if chunk>5:break
            scored_samples = scored_samples_origin.groupby('prompt').head((chunk+1)*n_of_bon).groupby('prompt').tail(n_of_bon)
            
            scored_samples = scored_samples.reset_index(drop=True)
            args.scored_samples = str(name_origin) + f'_{chunk}'
            make_comparison_data(scored_samples,args)
    else:
        make_comparison_data(scored_samples,args)
        
main()