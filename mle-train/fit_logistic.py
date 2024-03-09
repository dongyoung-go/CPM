import argparse
import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  GridSearchCV
import numpy as np
import re
import random
from pathlib import Path
import dill as pickle
import warnings
from transformers import AutoTokenizer
warnings.filterwarnings("ignore")


def train_model_g(dataset, model_name, feature_names):
    feature_cols = sum([[f'{feature}_{option}' for feature in feature_names] for option in ['A','B']],[])
    feature_values = dataset[feature_cols].astype(float).values
    feature_values_A = feature_values[:,:len(feature_names)]
    feature_values_B = feature_values[:,len(feature_names):]
    concat_feature_values = np.concatenate([feature_values_A,feature_values_B])
    feature_means = concat_feature_values.mean(0)
    feature_stds = concat_feature_values.std(0)+1e-7
    concat_feature_values -= feature_means
    concat_feature_values /= (feature_stds)
    feature_values_A = concat_feature_values[:len(feature_values)]
    feature_values_B = concat_feature_values[len(feature_values):]
    feature_values = (feature_values_A-feature_values_B)
    
    pipeline = Pipeline(steps=[('classifier', LogisticRegression(fit_intercept=False,random_state=42))])
    param_grid = {
        'classifier__penalty': ['l1','l2'],
        'classifier__C': np.logspace(-5, 5, 12),
        'classifier__solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=8, scoring='accuracy', n_jobs=-1)
    grid_search.fit(feature_values, dataset['labels'])
    print(f"{model_name} Best hyperparameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    print(f"{model_name}  accuracy: {best_accuracy}")
    best_model.mean = feature_means
    best_model.std = feature_stds
    return best_model,best_accuracy

def train_model_f(dataset, model_name, feature_names):
    features = [f'{feature_name}_A' for feature_name in feature_names] + [f'{feature_name}_B' for feature_name in feature_names]
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression(fit_intercept=False,random_state=42))])
    param_grid = {
        'classifier__penalty': ['l1','l2'],
        'classifier__C': np.logspace(-5, 5, 12),
        'classifier__solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=8, scoring='accuracy', n_jobs=-1)
    grid_search.fit(dataset[features], dataset['labels'])
    print(f"{model_name} Best hyperparameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    print(f"{model_name}  accuracy: {best_accuracy}")
    # joblib.dump(best_model, f'{model_name}.joblib')
    return best_model,best_accuracy

def recalibrate_and_scores(values, weights, means, std):
    normalized_values = (values - means) / std
    return np.dot(normalized_values, weights)

def get_model_statistics(model,ds, feature_names):
    try:
        # case f
        weights = np.mean([model.named_steps.classifier.coef_[0,:len(feature_names)],-model.named_steps.classifier.coef_[0,len(feature_names):]],axis=0)
        means = model.named_steps.scaler.mean_.reshape(2,-1).T.mean(1)
        stds = model.named_steps.scaler.scale_.reshape(2,-1).T.mean(1)
    except:
        # case g
        weights = model.named_steps.classifier.coef_.flatten()
        means = model.mean
        stds = model.std
    
    print('Model weights:', weights)
    print('Model means:', means)
    print('Model stds:', stds)
    
    # inference
    inference_dict = dict()
    features = [f'{feature_name}_A' for feature_name in feature_names] + [f'{feature_name}_B' for feature_name in feature_names]
    for inference_subset in ['train','test']:
        inference_dict[inference_subset] = {'chosen':[],'rejected':[]}

        df = ds[inference_subset].to_pandas()
        labels = df['labels'].values

        option1_values = df[features[:len(feature_names)]]
        option1_score = recalibrate_and_scores(option1_values, weights, means, stds)

        option2_values = df[features[len(feature_names):]]
        option2_score = recalibrate_and_scores(option2_values, weights, means, stds)

        for score1, score2, label in zip(option1_score,option2_score,labels):
            chosen = score1 if label==1 else score2
            rejected = score2 if label==1 else score1
            inference_dict[inference_subset]['chosen'].append(chosen)
            inference_dict[inference_subset]['rejected'].append(rejected)

        inference_dict[inference_subset]['option1']=option1_score
        inference_dict[inference_subset]['option2']=option2_score

    print(np.mean(np.array(inference_dict['train']['chosen'])>np.array(inference_dict['train']['rejected'])))
    print(np.mean(np.array(inference_dict['test']['chosen'])>np.array(inference_dict['test']['rejected'])))
    
    return weights,means,stds,inference_dict

def preprocess_hh_dataset(example,options=['A','B']):
    chosen_col = [x for x in example.keys() if x.endswith("_chosen")]
    rejected_col = [x for x in example.keys() if x.endswith("_rejected")]
    
    pattern1 = r'(_chosen)\b'
    pattern2 = r'(_rejected)\b'
    if random.random()<0.5:
        # chosen -> A
        example['labels']=1
        for col in chosen_col:    
            example[re.sub(pattern1, r'_A', col)] = example[col]
        for col in rejected_col:
            example[re.sub(pattern2, r'_B', col)] = example[col]
    else:
        # chosen -> B
        example['labels']=0
        for col in chosen_col:    
            example[re.sub(pattern1, r'_B', col)] = example[col]
        for col in rejected_col:
            example[re.sub(pattern2, r'_A', col)] = example[col]
    return example




tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
token_len = lambda response, prompt: (tokenizer(response, return_tensors="pt", add_special_tokens=True)['input_ids']).shape[1]

FEATURES = ['helpfulness', 'specificity','intent', 'factuality', 'easy-to-understand', 'relevance', 'readability', 'enough-detail', 'biased:', 'fail-to-consider-individual-preferences', 'repetetive', 'fail-to-consider-context', 'too-long']
FEATURE_CALLBACKS = {
        'length': lambda response, prompt: len(response)/100
        # 'token_length': token_len
}
# FEATURE_CALLBACKS = dict()
FEATURES.extend(FEATURE_CALLBACKS.keys())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_type', default='dongyoung4091/shp_with_features_20k_flan_t5_large')
    ap.add_argument('--shp_hh_dataset_type', nargs='+', default=[])
    ap.add_argument('--options', nargs='+', default=['A', 'B'])
    ap.add_argument('--context', default='history')
    ap.add_argument('--response-prefix', default='human_ref')
    ap.add_argument('--split', default='train',choices=['train','test'])
    ap.add_argument('--zeroshot_feature', action='store_true', help='use zeroshot feature extractor instead')
    ap.add_argument('--zeroshot_LM_type', default='xl')
    ap.add_argument('--g_fn', action='store_true', help='g use difference feature, f use concated feature')
    ap.add_argument('--short_feature', action='store_true', help='use only 3 features, readability, enough datil, failed consider context')
    ap.add_argument('--pm_pseudo_label', action='store_true', help='use PM pseudo label')
    ap.add_argument('--external_pm_pseudo_label', action='store_true', help='use PM pseudo label')
    ap.add_argument('--external_pm_pseudo_label2', action='store_true', help='use PM pseudo label') # just legacy
    ap.add_argument('--test', action='store_true', help='test')
    ap.add_argument('--subset_fit', default=0,type=int, help='to fit on small subsets')
    args = ap.parse_args()
    suffix = ''
        
    if len(args.shp_hh_dataset_type) == 0:
        args.shp_hh_dataset_type = [args.dataset_type]
        
    ################ Data Load ################
    ds_list = []
    for dataset_idx, dataset_type in enumerate(args.shp_hh_dataset_type):
        print(f'Load dataset {dataset_type}')
        
        args.dataset_type = dataset_type
        ds = datasets.load_dataset(args.dataset_type)
        if 'hh' in args.dataset_type:
            args.context='human'
            args.response_prefix='assistant'
        elif 'shp' in args.dataset_type:
            args.context = 'history'
            args.response_prefix='human_ref'
        feature_names = FEATURES
            
        if args.zeroshot_feature:
            if 'hh' in args.dataset_type:
                assert 'shp' not in args.dataset_type, 'Currently, extended features are only prepared for hh-rlhf dataset'
                ds_logprobs = ds
                model_size = '' if args.zeroshot_LM_type=='xl'  else f'{args.zeroshot_LM_type}_'
                args.dataset_type = f'dongyoung4091/hh-rlhf_with_features_flan_t5_large_flan_t5_{model_size}zeroshot' # zero-shot using flan-t5-xl
                print(f"Load new zeroshot feature, {args.dataset_type}")
                ds = datasets.load_dataset(args.dataset_type)
                if 'log_score_chosen' not in ds['train'].features:
                    ds['train'] = ds['train'].add_column('log_score_chosen',ds_logprobs['train']['log_score_chosen']) 
                    ds['train'] = ds['train'].add_column('log_score_rejected', ds_logprobs['train']['log_score_rejected'])
                    ds['test'] = ds['test'].add_column('log_score_chosen', ds_logprobs['test']['log_score_chosen'])
                    ds['test'] = ds['test'].add_column('log_score_rejected', ds_logprobs['test']['log_score_rejected'])
            elif 'shp' in args.dataset_type:
                ds_logprobs = ds
                # args.dataset_type = 'dongyoung4091/hh-rlhf_with_features_rx_reformatted' # zero-shot using sileod/deberta-v3-base-tasksource-nli
                model_size = '' if args.zeroshot_LM_type=='xl'  else f'{args.zeroshot_LM_type}_'
                args.dataset_type = f'dongyoung4091/shp_with_features_20k_flan_t5_large_flan_t5_{model_size}zeroshot' # zero-shot using flan-t5-xl
                print(f"Load new zeroshot feature, {args.dataset_type}")
                ds = datasets.load_dataset(args.dataset_type)
                if 'log_score_A' not in ds['train'].features:
                    ds['train'] = ds['train'].add_column('log_score_A',ds_logprobs['train']['log_score_A']) 
                    ds['train'] = ds['train'].add_column('log_score_B', ds_logprobs['train']['log_score_B'])
                    ds['test'] = ds['test'].add_column('log_score_A', ds_logprobs['test']['log_score_A'])
                    ds['test'] = ds['test'].add_column('log_score_B', ds_logprobs['test']['log_score_B'])            
            feature_names = ['zeroshot_' + x if x not in FEATURE_CALLBACKS else x for x in FEATURES ]
            suffix += '_zeroshot'
            if args.zeroshot_LM_type!='xl':
                suffix += f"_{args.zeroshot_LM_type}"
        if args.short_feature:
            feature_names = [ x for x in feature_names if any([feature in x for feature in ['readability', 'enough-detail', 'fail-to-consider-context']])]
            suffix += '_short'
        train_model = train_model_f 
        if args.g_fn:
            train_model = train_model_g
            suffix += '_g'
        if (args.external_pm_pseudo_label or args.external_pm_pseudo_label2) and dataset_idx==0: # We assume for second dataset we have better quality!
            if ('hh' in args.dataset_type) and (args.external_pm_pseudo_label):
                suffix += '_external_pm_pseudo_label' # just legacy
                def make_model_predict(input_dict):
                    winner_score = input_dict[args.split]['external_rm1_chosen']
                    loser_score = input_dict[args.split]['external_rm1_rejected']
                    model_predict = ['correct' if w>l else 'wrong' for w,l in zip(winner_score,loser_score)]
                    return model_predict
                def change_winner_loser(example):
                    original_chosen_col = [col for col in example.keys() if 'chosen' in col]
                    original_rejected_col = [col for col in example.keys() if 'rejected' in col]
                    original_chosen_value = [example[col] for col in original_chosen_col]
                    original_rejected_value = [example[col] for col in original_rejected_col]
                    if example['model_predict']=='wrong':
                        for col, value in zip(original_rejected_col,original_chosen_value):
                            example[col] = value
                        for col, value in zip(original_chosen_col,original_rejected_value):
                            example[col] = value
                    return example
                ds_external = datasets.load_dataset('dongyoung4091/hh-rlhf_with_features_rx_reformatted')
                if 'external_rm1_chosen' not in ds['train'].features:
                    ds['train'] = ds['train'].add_column('external_rm1_chosen',ds_external['train']['external_rm1_chosen']) 
                    ds['train'] = ds['train'].add_column('external_rm1_rejected', ds_external['train']['external_rm1_rejected'])
                    ds['test'] = ds['test'].add_column('external_rm1_chosen', ds_external['test']['external_rm1_chosen'])
                    ds['test'] = ds['test'].add_column('external_rm1_rejected', ds_external['test']['external_rm1_rejected'])
            elif ('shp' in args.dataset_type) and (args.external_pm_pseudo_label2):
                suffix += '_external_pm_pseudo_label2'# just legacy
                def make_model_predict(input_dict):
                    A_score = input_dict[args.split]['external_rm1_A']
                    B_score = input_dict[args.split]['external_rm1_B']
                    org_labels = input_dict[args.split]['labels']
                    model_predict = []
                    for a,b,l in zip(A_score,B_score,org_labels):
                        if l==1:
                            model_predict.append('correct' if (a>b) else 'wrong')
                        elif l==0:
                            model_predict.append('correct' if (a<b) else 'wrong')
                    return model_predict
                def change_winner_loser(example):
                    original_A_col = [col for col in example.keys() if '_A' in col]
                    original_B_col = [col for col in example.keys() if '_B' in col]
                    original_A_value = [example[col] for col in original_A_col]
                    original_B_value = [example[col] for col in original_B_col]
                    if example['model_predict']=='wrong':
                        for col, value in zip(original_B_col,original_A_value):
                            example[col] = value
                        for col, value in zip(original_A_col,original_B_value):
                            example[col] = value
                    return example
                if args.external_pm_pseudo_label2:
                    ds_external = datasets.load_from_disk('../mle-train/data/shp_with_features_20k_flan_t5_large_sileod')
                    print('Load slieod')
                if 'external_rm1_A' not in ds['train'].features:
                    ds['train'] = ds['train'].add_column('external_rm1_A',ds_external['train']['external_rm1_A']) 
                    ds['train'] = ds['train'].add_column('external_rm1_B', ds_external['train']['external_rm1_B'])
                    ds['test'] = ds['test'].add_column('external_rm1_A', ds_external['test']['external_rm1_A'])
                    ds['test'] = ds['test'].add_column('external_rm1_B', ds_external['test']['external_rm1_B'])
            model_predict = make_model_predict(ds)
            print('mean model_predict',np.mean(np.array(model_predict)=='correct'))
            ds[args.split] = ds[args.split].add_column('model_predict', model_predict)
            ds[args.split] = ds[args.split].map(change_winner_loser)
            ds[args.split] = ds[args.split].remove_columns('model_predict')
            
            
        if args.pm_pseudo_label:
            suffix += '_PM_pseudo_label'
            if 'shp' in args.dataset_type:
                def make_model_predict(input_dict):
                    if 'winner' in input_dict[args.split]:
                        winner_score = input_dict[args.split]['winner']
                        loser_score = input_dict[args.split]['loser']
                    else:
                        winner_score = input_dict[args.split]['chosen']
                        loser_score = input_dict[args.split]['rejected']
                    model_predict = ['correct' if w>l else 'wrong' for w,l in zip(winner_score,loser_score)]
                    return model_predict
                def change_winner_loser(example):
                    original_A_col = [col for col in example.keys() if '_A' in col]
                    original_B_col = [col for col in example.keys() if '_B' in col]
                    original_A_value = [example[col] for col in original_A_col]
                    original_B_value = [example[col] for col in original_B_col]
                    if example['model_predict']=='wrong':
                        for col, value in zip(original_B_col,original_A_value):
                            example[col] = value
                        for col, value in zip(original_A_col,original_B_value):
                            example[col] = value
                    return example

                unseen_data = "_test" if args.split== "train" else ""
                pl_dataset = f'../reward_model/model_ckpt/flan-t5-xl_dongyoung4091_shp_with_features_20k_lr1e-05{unseen_data}_shp-prepend_split_1234_output.pickle'
                print(f"Load pseudo label using {pl_dataset}")
                with open(pl_dataset, "rb") as pickle_file:
                    output_dict =pickle.load(pickle_file)
            elif 'hh' in args.dataset_type:
                def make_model_predict(input_dict):
                    if 'winner' in input_dict[args.split]:
                        winner_score = input_dict[args.split]['winner']
                        loser_score = input_dict[args.split]['loser']
                    else:
                        winner_score = input_dict[args.split]['chosen']
                        loser_score = input_dict[args.split]['rejected']
                    model_predict = ['correct' if w>l else 'wrong' for w,l in zip(winner_score,loser_score)]
                    return model_predict
                def change_winner_loser(example):
                    original_chosen_col = [col for col in example.keys() if 'chosen' in col]
                    original_rejected_col = [col for col in example.keys() if 'rejected' in col]
                    original_chosen_value = [example[col] for col in original_chosen_col]
                    original_rejected_value = [example[col] for col in original_rejected_col]
                    if example['model_predict']=='wrong':
                        for col, value in zip(original_rejected_col,original_chosen_value):
                            example[col] = value
                        for col, value in zip(original_chosen_col,original_rejected_value):
                            example[col] = value
                    return example
                unseen_data = "_test" if args.split== "train" else ""
                pl_dataset = f'../reward_model/model_ckpt/flan-t5-xl_dongyoung4091_hh-rlhf_with_features_lr1e-05{unseen_data}_prepend_split_1234_output.pickle'
                print(f"Load pseudo label using {pl_dataset}")
                with open(pl_dataset, "rb") as pickle_file:
                    output_dict =pickle.load(pickle_file)
            model_predict = make_model_predict(output_dict)
            print('mean model_predict',np.mean(np.array(model_predict)=='correct'))
            ds[args.split] = ds[args.split].add_column('model_predict', model_predict)
            ds[args.split] = ds[args.split].map(change_winner_loser)
            ds[args.split] = ds[args.split].remove_columns('model_predict')
            

        short_data_name = args.dataset_type.split('/')[-1] + ('_concated_ds' if len(args.shp_hh_dataset_type)>1 else '')
        if 'hh' in args.dataset_type:
            ds = ds.map(preprocess_hh_dataset)
            original_col = [x for x in ds[args.split].column_names if x.endswith('_chosen') or x.endswith('_rejected') ]
            ds = ds.remove_columns(original_col)
        if args.test:
            suffix += '_TEST_'
        if args.subset_fit>0:
            suffix += f'_subset_{args.subset_fit}'
            ds['train'] = ds['train'].shuffle(seed=100).select(range(args.subset_fit))
            ds['test'] = ds['test'].shuffle(seed=100).select(range(args.subset_fit))
            
            
        for feature, callback_fn in FEATURE_CALLBACKS.items():
            def apply_callback(row):
                for option in args.options:
                    row[f'{feature}_{option}'] = callback_fn(row[f'{args.response_prefix}_{option}'],row[args.context]) 
                return row
            ds = ds.map(apply_callback) 
        ds_list.append(ds)
        
    # Concat ds_list
    for idx in range(len(ds_list)):
        ds = ds_list[idx]
        if 'human_ref_A' in ds['train'].features:
            LENGTH_CUT = 1e8
            print('Apply Filter')
            ds = ds.filter(lambda x: len(x["human_ref_A"]) < LENGTH_CUT and len(x["human_ref_B"]) < LENGTH_CUT)
        print(ds)
        ds = ds.remove_columns([feature for feature in ds['train'].features if any([targ_feat in feature for targ_feat in feature_names+['labels']]) is False])
        for split in ['train','test']:
            new_features = ds[split].features.copy()
            for col in ds[split].features:
                if any([targ_feat in col for targ_feat in feature_names]):
                    new_features[col] = datasets.Value("double")
            ds[split] = ds[split].cast(new_features)
        ds_list[idx] = ds
        
    fin_ds = dict()
    fin_ds['train'] = datasets.concatenate_datasets([ds['train'] for ds in ds_list])
    fin_ds['test'] = datasets.concatenate_datasets([ds['test'] for ds in ds_list])
    ds = datasets.DatasetDict(fin_ds)
        
        
    ################ Fit Model ################
    # fit model
    gold_model,model_acc = train_model(ds[args.split].to_pandas(), args.split, feature_names)
    # save model
    save_fname = f'{short_data_name}_{args.split}_logistic{suffix}'

    # get model coeffs
    weights,means,stds,inference_dict=get_model_statistics(gold_model, ds, feature_names)
    rewards = np.concatenate([inference_dict[args.split]['chosen'],inference_dict[args.split]['rejected']])

    output_path = Path(save_fname).stem+'.pkl'
    pickle.dump(
            {'feature_names': feature_names,
                'feature_callbacks': FEATURE_CALLBACKS,
                'features_mean': means,
                'features_std': stds,
                'coefficients': weights,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),         
                'acc':model_acc},
            open('./out/'+output_path, 'wb'))
    print(f'saved to {output_path}')

    output_save_fname = f'{Path(save_fname).stem}_output.pkl'
    with open('./out/'+output_save_fname, "wb") as pickle_file:
        pickle.dump(inference_dict, pickle_file)
    print(f'saved to {output_save_fname}')
    
    # if args.pm_pseudo_label or args.external_pm_pseudo_label or args.test:
    print('Finish without making second proxy model')
    return None
    
main()