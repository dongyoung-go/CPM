import sys
import argparse
import math
import random
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets,load_from_disk
import transformers
import os


sys.path.append('../')
from utils.constant import FEATURES
from rm_dataset import PairwiseDataset, preprocess_shp_dataset, preprocess_hh_dataset

print(transformers.__version__)

parser = argparse.ArgumentParser()

# environment
parser.add_argument('--seed', dest='seed', type=int, default=1234)
parser.add_argument('--session_type', dest='session_type', type=str, default='pretrain')
parser.add_argument('--reschedule', dest='reschedule', type=int, default=1)

# basic settings
parser.add_argument('--load_path', dest='load_path', type=str, default=False)
parser.add_argument('--log_freq', dest='log_freq', type=int, default=1_000)
parser.add_argument('--eval_freq', dest='eval_freq', type=int, default=3_000)
parser.add_argument('--save_freq', dest='save_freq', type=int, default=3_000)
parser.add_argument('--num_workers', dest='num_workers', type=int, default=16)
parser.add_argument('--debugging', action='store_true',default=False)

# model hyperparams
parser.add_argument('--lr', dest='lr', type=float, default=1e-5, help='learning rate of adam')
parser.add_argument('--max_norm', dest='max_norm', type=float, default=0.01, help='gradient max_norm')
parser.add_argument('--vocab_size', dest='vocab_size', type=int, default=50261)
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.01)

# training strategies
parser.add_argument('--micro_batch_size', dest='micro_batch_size', type=int, default=8, help='number of batch_size')
parser.add_argument('--global_batch_size', dest='global_batch_size', type=int, default=16)
parser.add_argument('--epochs', dest='epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--warmup', dest='warmup', type=int, default=100, help='number of steps for warmup')
parser.add_argument('--model_name', dest='model_name', type=str, default='google/flan-t5-xl')
parser.add_argument('--dataset_type', dest='dataset_type', type=str, default='dongyoung4091/hh-rlhf_with_features')
parser.add_argument('--save_model',   action='store_true',default=False)
parser.add_argument('--subset', dest='subset', type=str, default='train')
parser.add_argument('--prepend_mode', dest='prepend_mode', type=str, default='prepend_split')
parser.add_argument('--pseudo_label',   action='store_true',default=False)
parser.add_argument('--loss_type', type=str, default='reward_model',choices=['reward_model','mse'])
parser.add_argument('--target_features', nargs='+', default=[], help='features in interest')
parser.add_argument('--pseudo_eval',   action='store_true',default=False)
parser.add_argument('--external_pseudo_label',   action='store_true',default=False, help='use external_RM1 as gold PM and use its pseudo_label')
parser.add_argument('--sileod_external_pseudo_label',   action='store_true',default=False, help='use external_RM1 as gold PM and use its pseudo_label')
parser.add_argument('--subset_fit', default=0,type=int, help='to fit on small subsets')


args = parser.parse_args()
print(args)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
global_batch_size = args.global_batch_size
load_batch_size = min(args.micro_batch_size, global_batch_size)
assert global_batch_size % load_batch_size == 0
steps_per_update = global_batch_size // load_batch_size
scaled_lr = args.lr * global_batch_size / 16

short_model_name = args.model_name.split("/")[-1]
SAVE_PATH = f'./model_ckpt/{short_model_name}_{args.dataset_type.replace("/","_")}_lr{args.lr}{"" if args.subset=="train" else "_"+args.subset}_{args.prepend_mode}_{args.seed}'
if args.pseudo_label:
    SAVE_PATH += '_pseudo_label'
if args.external_pseudo_label:
    SAVE_PATH += '_external_pseudo_label'
if args.sileod_external_pseudo_label:
    SAVE_PATH += '_sileod_external_pseudo_label'
if args.subset_fit>0:
    SAVE_PATH += f'_subset_{args.subset_fit}'

# set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dim = 1 # scalar, by default
if args.target_features:
    if args.target_features ==['all']:
        target_columns = [x for x in FEATURES.keys() if (x!='length')]
    else:
        target_columns = [x for x in FEATURES.keys() if x in args.target_features]
    output_dim = len(target_columns)
    SAVE_PATH += f'_dim{output_dim}'
    
if os.path.exists(SAVE_PATH):
    raise FileExistsError(f"'{SAVE_PATH}' already exists.")    

print(f'SAVE_PATH: {SAVE_PATH}')
    
    
## Load model
model_name = args.model_name

# Load the pretrained T5 model and tokenizer
# https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/models/t5/modeling_t5.py#L1748
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
linear = torch.nn.Linear(model.config.hidden_size, output_dim)
model.set_output_embeddings(linear)
model.to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

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
elif args.subset=='test':
    print("Swap train & test set!")
    train_data = dataset['test']
    test_data = dataset['train']
elif args.subset=='all':
    # Concatenate train and test datasets
    train_data = concatenate_datasets([dataset['train'], 
                                       Dataset.from_dict(dataset['test'][:-1000])])
    test_data = Dataset.from_dict(dataset['test'][-1000:])
    

if args.pseudo_label or args.external_pseudo_label:
    def change_winner_loser(example):
        original_chosen = example['chosen']
        original_rejected = example['rejected']
        if example['model_predict']=='wrong':
            example['chosen'] = original_rejected
            example['rejected'] = original_chosen
        return example
    def make_model_predict(input_dict,label=None,subset=args.subset):
        if label is not None:
            winner_label = label[0]
            loser_label = label[1]
        else:
            if 'winner' in input_dict:
                winner_label = 'winner'
                loser_label = 'loser'
            else:
                winner_label = 'chosen'
                loser_label = 'rejected'
        winner_score = input_dict[subset][winner_label]
        loser_score = input_dict[subset][loser_label]
        model_predict = ['correct' if w>l else 'wrong' for w,l in zip(winner_score,loser_score)]
        return model_predict
    
    if 'shp' in args.dataset_type:
        print('<Original chosen>')
        for idx in range(10):
            print(train_data[idx]['chosen'])
        
        # load pseudo_label dataset for training
        if args.sileod_external_pseudo_label:
            gold_trained_data = 'train' if args.subset=='test' else 'test'
            ds = load_from_disk('../mle-train/data/shp_with_features_20k_flan_t5_large_sileod')        
            ds = ds.map(lambda x:preprocess_shp_dataset(x,prepend_mode=args.prepend_mode)) # 'hh-rlhf'
            model_predict = make_model_predict(ds,('external_rm1_chosen','external_rm1_rejected'),subset=args.subset)
            train_data = train_data.add_column('model_predict', model_predict)
            train_data = train_data.map(change_winner_loser)
            model_predict = make_model_predict(ds,('external_rm1_chosen','external_rm1_rejected'),subset=gold_trained_data)
            test_data = test_data.add_column('model_predict', model_predict)
            test_data = test_data.map(change_winner_loser)
        else:
            gold_trained_data = 'train' if args.subset=='test' else 'test'
            if 'xl' in args.model_name:
                # legacy
                pl_dataset = f'./notebook/shp_with_features_20k_{args.subset}_pseudo_labels.feather'
                print(f'Replacing training dataset with {pl_dataset}!')
                pl_dataset = pd.read_feather(pl_dataset)
                pl_dataset = Dataset.from_pandas(pl_dataset)
                # Replace original labels with pseudo_labels
                pl_dataset = pl_dataset.remove_columns(["labels"])
                pl_dataset = pl_dataset.rename_column("pseudo_labels", "labels")
                pl_dataset = pl_dataset.map(lambda x:preprocess_shp_dataset(x,prepend_mode=args.prepend_mode))
                pl_dataset = pl_dataset.remove_columns(["human_ref_A", "human_ref_B"])
                train_data = pl_dataset
                
                pl_dataset = f'./notebook/shp_with_features_20k_{gold_trained_data}_pseudo_labels.feather'
                pl_dataset = pd.read_feather(pl_dataset)
                pl_dataset = Dataset.from_pandas(pl_dataset)
                # Replace original labels with pseudo_labels
                pl_dataset = pl_dataset.remove_columns(["labels"])
                pl_dataset = pl_dataset.rename_column("pseudo_labels", "labels")
                pl_dataset = pl_dataset.map(lambda x:preprocess_shp_dataset(x,prepend_mode=args.prepend_mode))
                pl_dataset = pl_dataset.remove_columns(["human_ref_A", "human_ref_B"])
                test_data = pl_dataset
            elif 'large' in args.model_name:
                pl_dataset = f'./model_ckpt/{short_model_name}_dongyoung4091_shp_with_features_20k_lr{args.lr}_{gold_trained_data}_shp-prepend_split_1234_output.pickle'
                with open(pl_dataset, "rb") as pickle_file:
                    output_dict =pickle.load(pickle_file)
                model_predict = make_model_predict(output_dict)
                train_data = train_data.add_column('model_predict', model_predict)
                train_data = train_data.map(change_winner_loser)

        print('<Pseudo chosen>')
        for idx in range(10):
            print(train_data[idx]['chosen'])
    elif 'hh-rlhf' in args.dataset_type:            
        if args.external_pseudo_label:
            ds = load_dataset('dongyoung4091/hh-rlhf_with_features_rx_reformatted')
            model_predict = make_model_predict(ds,('external_rm1_chosen','external_rm1_rejected'))
            train_data = train_data.add_column('model_predict', model_predict)
            train_data = train_data.map(change_winner_loser)
        else:
            gold_trained_data = 'train' if args.subset=='test' else 'test'
            pl_dataset = f'./model_ckpt/{short_model_name}_dongyoung4091_hh-rlhf_with_features_lr{args.lr}_{gold_trained_data}_prepend_split_1234_output.pickle'
            with open(pl_dataset, "rb") as pickle_file:
                output_dict =pickle.load(pickle_file)
            model_predict = make_model_predict(output_dict)
            train_data = train_data.add_column('model_predict', model_predict)
            train_data = train_data.map(change_winner_loser)

        winner_score = output_dict[gold_trained_data]['winner']
        loser_score = output_dict[gold_trained_data]['loser']
        model_predict = ['chosen' if w>l else 'rejected' for w,l in zip(winner_score,loser_score)]
        test_data = test_data.add_column('model_predict', model_predict)
        test_data = test_data.map(change_winner_loser)
    
# Create the datasets and dataloaders
print("<Data Samples>")
if 'prepend' in train_data[0]:
    print(f"<Prepend>\n{train_data[0]['prepend']}")
print(f"<Chosen>\n{train_data[0]['chosen']}")
print(f"<Rejected>\n{train_data[0]['rejected']}")

if args.subset_fit>0:
    train_data = train_data.shuffle(seed=100).select(range(args.subset_fit))
    test_data = test_data.shuffle(seed=100).select(range(args.subset_fit))
    
train_dataset = PairwiseDataset(train_data, tokenizer)
test_dataset = PairwiseDataset(test_data, tokenizer)

if args.target_features:
    train_dataset.target_columns = target_columns
    test_dataset.target_columns = train_dataset.target_columns
    if 'shp' in args.dataset_type:
        train_dataset.options = ['A','B']
    elif 'hh-rlhf' in args.dataset_type:
        train_dataset.options = ['chosen','rejected']
    test_dataset.options = train_dataset.options
    

train_dataloader = DataLoader(train_dataset, batch_size=args.micro_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.micro_batch_size, shuffle=True)

## optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
total_epoch = len(train_dataloader) * args.epochs
if args.subset_fit>0:
    divide_cnt = (10_000/args.subset_fit)
    args.warmup = max(20,args.warmup/divide_cnt)
    args.eval_freq = min(args.subset_fit,1000)
    args.save_freq = min(args.subset_fit,1000)
lr_func = lambda epoch: min((epoch + 1) / (args.warmup + 1e-8), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=False)

## Start training
loss_list = []
eval_list = []
steps = 0
best_acc = 0
patience = 0
if args.loss_type=='mse':
    best_acc = 1e3 # arbitrary large number
args.eval_freq = min(args.eval_freq,len(train_dataloader))
args.save_freq = min(args.save_freq,len(train_dataloader))

def evaluate(model,test_dataloader):
    model.eval()
    with torch.no_grad():
        ev_list_ = []
        ev_idx = 0
        for kwargs in tqdm(test_dataloader,desc='Evaluating'):
            ev_idx += 1
            if ev_idx>1000:break
            chosen_ids = kwargs['chosen_ids'].squeeze(1).to(device)
            rejected_ids = kwargs['rejected_ids'].squeeze(1).to(device)
            if 'prepend_ids' in kwargs:
                prepend_ids = kwargs['prepend_ids'].squeeze(1).to(device)
                reward_w = model(input_ids=prepend_ids, decoder_input_ids=chosen_ids).logits[:,:,:]
                reward_l = model(input_ids=prepend_ids, decoder_input_ids=rejected_ids).logits[:,:,:]
            else:
                reward_w = model(input_ids=chosen_ids, decoder_input_ids=chosen_ids).logits[:,:,:]
                reward_l = model(input_ids=rejected_ids, decoder_input_ids=rejected_ids).logits[:,:,:]

            reward_w_last = reward_w[:,-1]
            reward_l_last = reward_l[:,-1]
            if args.loss_type=='reward_model':
                ev_list_.append((reward_w_last>reward_l_last).float().mean().item())
            elif args.loss_type=='mse':
                mse_error_ = (reward_w_last - kwargs['chosen_value'].to(device)).pow(2).mean().item()
                mse_error = mse_error_ + (reward_l_last - kwargs['rejected_value'].to(device)).pow(2).mean().item()
                ev_list_.append(mse_error)
    return np.mean(ev_list_)

eval_acc = evaluate(model,test_dataloader)
print(f'initial eval_acc: {eval_acc}')
best_acc = eval_acc
eval_list.append(eval_acc)

for epoch in range(args.epochs):
    model.train()
    for kwargs in tqdm(train_dataloader,desc='Training'):
        chosen_ids = kwargs['chosen_ids'].squeeze(1).to(device)
        rejected_ids = kwargs['rejected_ids'].squeeze(1).to(device)
        if 'prepend_ids' in kwargs:
            prepend_ids = kwargs['prepend_ids'].squeeze(1).to(device)
            reward_w = model(input_ids=prepend_ids, decoder_input_ids=chosen_ids).logits[:,:,:]
            reward_l = model(input_ids=prepend_ids, decoder_input_ids=rejected_ids).logits[:,:,:]
        else:
            output = model(input_ids=chosen_ids, decoder_input_ids=chosen_ids).logits
            print('output.size()',output.size())
            reward_w = output[:,:,0]
            print('reward_w',reward_w)
            reward_l = model(input_ids=rejected_ids, decoder_input_ids=rejected_ids).logits[:,:,:]
            
        # get last_hidden_state w last token
        reward_w_last = reward_w[:,-1] ## last_hidden_state
        reward_l_last = reward_l[:,-1] ## last_hidden_state

        reward_w_last = reward_w_last.to(torch.float32)
        reward_l_last = reward_l_last.to(torch.float32)
        if args.debugging:
            print('chosen_ids.size()',chosen_ids.size())
            print('rejected_ids.size()',rejected_ids.size())
            if 'prepend_ids' in kwargs:
                print('prepend_ids.size()',prepend_ids.size())
                print('< prepend_ids >')
                print(tokenizer.decode(prepend_ids[0],skip_special_tokens=True))
            print('< Chosen >')
            print(tokenizer.decode(chosen_ids[0],skip_special_tokens=True))
            print('< Rejected >')
            print(tokenizer.decode(rejected_ids[0],skip_special_tokens=True))
            print('reward_w_last',reward_w_last)
            print('reward_l_last',reward_l_last)
            raise ValueError ## Terminate
        if args.loss_type=='reward_model':
            loss_ = -1*torch.nn.functional.logsigmoid(reward_w_last - reward_l_last)
            loss = loss_.mean()
        elif args.loss_type=='mse':
            loss_ = (reward_w_last - kwargs['chosen_value'].to(device)).pow(2).mean()
            loss = loss_ + (reward_l_last - kwargs['rejected_value'].to(device)).pow(2).mean()
        loss_list.append(loss.item())
        loss.backward()
        if steps % steps_per_update == 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        
        steps += 1
        if steps % args.eval_freq==0:
            eval_acc = evaluate(model,test_dataloader)
            print(f'eval_acc: {eval_acc}')
            eval_list.append(eval_acc)
            
        if (steps % args.save_freq == 0)|(steps == total_epoch):
            if args.loss_type=='reward_model':
                if best_acc<eval_acc:
                    best_acc = eval_acc
                    patience = 0
                else:
                    patience += 1
                    if patience>4:
                        print(f'Early Stop with patience{patience}')
                        raise ValueError
                    continue
            elif args.loss_type=='mse':
                if best_acc>eval_acc:
                    best_acc = eval_acc
                else:
                    continue
                
            if args.save_model:
                torch.save({
                            'model':model.state_dict(),
                            # 'optimizer':optimizer.state_dict(),
                            # 'scheduler': scheduler.state_dict(),
                            'loss_list':loss_list,
                            'eval_list':eval_list,
                            'epoch':epoch,
                            'args':vars(args),
                            }, SAVE_PATH)
            else:
                torch.save({
                            'loss_list':loss_list,
                            'eval_list':eval_list,
                            'epoch':epoch,
                            'args':vars(args),
                            }, SAVE_PATH)
            print(f'Model saved to {SAVE_PATH}')