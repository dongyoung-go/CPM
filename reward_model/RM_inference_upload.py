import pickle
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset

def preprocess_output(model_pickle):
    # Concat r_chosen and r_rejected to get statistics
    model_A_train_output = sum(model_pickle['train'].values(),[])
    model_A_test_output = sum(model_pickle['test'].values(),[])

    print(f'Model A on dataset[\'train\']\nmean: {np.mean(model_A_train_output):0.3f}, std: {np.std(model_A_train_output):0.3f}')
    print(f'Model A on dataset[\'test\']\nmean: {np.mean(model_A_test_output):0.3f}, std: {np.std(model_A_test_output):0.3f}')

    # Subtract train_averge to bias
    model_A_bias = np.mean(model_A_train_output)
    model_pickle_debiased = {subset: {option: reward - model_A_bias for option, reward in rewards.items()} for subset,rewards in  model_pickle.items()}

    for subset in model_pickle_debiased.keys():
        print(f"{subset} accuracy: {np.mean(model_pickle_debiased[subset]['chosen']>model_pickle_debiased[subset]['rejected'])}")

    return model_pickle_debiased

# Load data
fname = './model_ckpt/flan-t5-xl_dongyoung4091_hh-rlhf_with_features_lr1e-05_test_prepend_split_1234_output.pickle'
with open(fname,'rb') as fr:
    model_B_pickle = pickle.load(fr)
    
fname = './model_ckpt/flan-t5-xl_dongyoung4091_hh-rlhf_with_features_lr1e-05_prepend_split_1234_output.pickle'
with open(fname,'rb') as fr:
    model_A_pickle = pickle.load(fr)

model_A_pickle = preprocess_output(model_A_pickle)
model_B_pickle = preprocess_output(model_B_pickle)

dataset = load_dataset('dongyoung4091/hh-rlhf_with_features')
if 'test' not in dataset.keys():
    dataset = dataset['train'].train_test_split(test_size=0.5, shuffle=False)

dataset['train'] = dataset['train'].add_column('model_A_chosen',model_A_pickle['train']['chosen'])
dataset['train'] = dataset['train'].add_column('model_A_rejected',model_A_pickle['train']['rejected'])
dataset['train'] = dataset['train'].add_column('model_B_chosen',model_B_pickle['test']['chosen'])
dataset['train'] = dataset['train'].add_column('model_B_rejected',model_B_pickle['test']['rejected'])

dataset['test'] = dataset['test'].add_column('model_A_chosen',model_A_pickle['test']['chosen'])
dataset['test'] = dataset['test'].add_column('model_A_rejected',model_A_pickle['test']['rejected'])
dataset['test'] = dataset['test'].add_column('model_B_chosen',model_B_pickle['train']['chosen'])
dataset['test'] = dataset['test'].add_column('model_B_rejected',model_B_pickle['train']['rejected'])

# dataset.push_to_hub('dongyoung4091/hh-rlhf_with_features_rx')