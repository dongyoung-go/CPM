from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
import sys
sys.path.append('../')
from utils.constant import FEATURES
from datasets import load_dataset
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import defaultdict
import pickle

ap = argparse.ArgumentParser()
ap.add_argument('--model_name', default='google/flan-t5-xl')
# ap.add_argument('--model-path', default='./model_ckpt/flan-t5-xl_dongyoung4091_hh-rlhf_with_features_lr1e-05_test_prepend_split_1234')

ap.add_argument('--dataset_type', dest='dataset_type', type=str, default='dongyoung4091/hh-rlhf_with_features')
ap.add_argument('--subset', dest='subset', type=str, default='test')
ap.add_argument('--prepend_mode', dest='prepend_mode', type=str, default='prepend_split')
ap.add_argument('--pseudo_label',   action='store_true',default=False)
ap.add_argument('--external_pseudo_label',   action='store_true',default=False)
ap.add_argument('--logistic_pseudo_label',   action='store_true',default=False)
ap.add_argument('--sileod_external_pseudo_label',   action='store_true',default=False)
ap.add_argument('--lr', dest='lr', type=str, default="1e-05", help='learning rate of adam')
ap.add_argument('--check_best',   action='store_true',default=False)
args = ap.parse_args()
args.model_path = f"./model_ckpt/{args.model_name.split('/')[1]}_{args.dataset_type.replace('/','_')}_lr{args.lr}{'_test' if args.subset=='test' else ''}_{args.prepend_mode}_1234"
if args.check_best: # Just to check what is "*_best"
    args.model_path += '_best'
if args.pseudo_label:
    args.model_path += '_pseudo_label'
if args.external_pseudo_label:
    args.model_path += '_external_pseudo_label'
if args.sileod_external_pseudo_label:
    args.model_path += '_sileod_external_pseudo_label'
if args.logistic_pseudo_label:
    args.model_path += '_logistic_pseudo_label'
    

print(f'SAVE_PATH: {args.model_path}')
print('args.model_path',args.model_path)

# set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained T5 model and tokenizer
model_name = args.model_name
model = T5ForConditionalGeneration.from_pretrained(model_name)
linear = torch.nn.Linear(model.config.hidden_size, 1)
model.set_output_embeddings(linear)
tokenizer = T5Tokenizer.from_pretrained(model_name)
# Load and map the check point
if os.path.isdir(args.model_path):
    args.model_path += '.pth'
ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()


#### Legacy code. Need modularize!
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from rm_dataset import PairwiseDataset, preprocess_shp_dataset, preprocess_hh_dataset

# # Load preference dataset
if args.dataset_type in ['harmless','helpfulness']:
    # https://huggingface.co/datasets/Anthropic/hh-rlhf#usage
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir=f"{args.dataset_type}-base")
elif 'shp' in args.dataset_type:
    # Just for the compatability, I changed human_ref_A/B to chosen/rejected
    dataset = load_dataset(args.dataset_type)
    dataset = dataset.map(lambda x:preprocess_shp_dataset(x,prepend_mode=args.prepend_mode)) # 'hh-rlhf'
    dataset = dataset.remove_columns(["human_ref_A", "human_ref_B"])    
elif 'hh-rlhf_with_features' in args.dataset_type:
    dataset = load_dataset(args.dataset_type)
    # as dongyoung4091/hh-rlhf_with_features has only training dataset with 20k rows
    if 'test' not in dataset.keys():
        # https://stackoverflow.com/questions/72499850/how-to-load-two-pandas-dataframe-into-hugginfaces-dataset-object
        dataset = dataset['train'].to_pandas()

        train_dataset = dataset[:len(dataset)//2]
        test_dataset = dataset[len(dataset)//2:]

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_dataset),
            "test": Dataset.from_pandas(test_dataset)
            })
    dataset = dataset.map(lambda x:preprocess_hh_dataset(x,prepend_mode=args.prepend_mode))

# Split the dataset into train and test
if args.subset=='train':
    train_data = dataset['train']
    test_data = dataset['test']
else:
    print("Swap train & test set!")
    train_data = dataset['test']
    test_data = dataset['train']
    
# Create the datasets and dataloaders
print("<Data Samples>")
if 'prepend' in train_data[0]:
    print(f"<Prepend>\n{train_data[0]['prepend']}")
print(f"<Chosen>\n{train_data[0]['chosen']}")
print(f"<Rejected>\n{train_data[0]['rejected']}")
train_dataset = PairwiseDataset(train_data, tokenizer)
test_dataset = PairwiseDataset(test_data, tokenizer)


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def model_inference(model,dataloader):
    output_dict = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for kwargs in tqdm(dataloader,desc='Evaluating'):
            chosen_ids = kwargs['chosen_ids'].squeeze(1).to(device)
            rejected_ids = kwargs['rejected_ids'].squeeze(1).to(device)
            ## TODO: Set prepend_ids as the only option, as counterpart isn't reasonable
            # with torch.cuda.amp.autocast():
            if 'prepend_ids' in kwargs:
                prepend_ids = kwargs['prepend_ids'].squeeze(1).to(device)
                reward_w = model(input_ids=prepend_ids, decoder_input_ids=chosen_ids).logits[:,:,:]
                reward_l = model(input_ids=prepend_ids, decoder_input_ids=rejected_ids).logits[:,:,:]
            else:
                reward_w = model(input_ids=chosen_ids, decoder_input_ids=chosen_ids).logits[:,:,:]
                reward_l = model(input_ids=rejected_ids, decoder_input_ids=rejected_ids).logits[:,:,:]

            # get last_hidden_state w last token
            reward_w_last = reward_w[:,-1]
            reward_l_last = reward_l[:,-1]

            output_dict['chosen'].append(reward_w_last.tolist())
            output_dict['rejected'].append(reward_l_last.tolist())
    output_dict = {k:sum(v,[]) for k,v in output_dict.items()}
    return output_dict

train_output = model_inference(model,train_dataloader)
test_output = model_inference(model,test_dataloader)
output_dict = {'train':train_output, 'test':test_output}

        
save_fname = f'{args.model_path}_output.pickle'
with open(save_fname, "wb") as pickle_file:
    pickle.dump(output_dict, pickle_file)
print(f'saved to {save_fname}')