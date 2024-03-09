import argparse
import pandas as pd
import torch
from tqdm import tqdm
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from collections import namedtuple
from pprint import pprint

# https://github.com/anonymous/anonymous/blob/main/anonymous/utils/helpers.py
def get_token_first_indices(x, token):
    """Find the first occurrence of a token in a 2D token array.
    Parameters
    ----------
    x: 2D int array
       list of token sequences
    token: int
        token to search
    Returns
    ------
    1D array containing the position of the first occurrence of the token or -1 if not found
    """
    if 0 == x.shape[-1]:
        return torch.tensor(-1).repeat(x.shape[0])
    else:
        mask = token == x
        mask_max_values, mask_max_indices = torch.max(mask, dim=1)
        mask_max_indices[mask_max_values == 0] = -1
        return mask_max_indices
    
# https://github.com/anonymous/anonymous/blob/main/anonymous/distributions/lm_distribution.py
TextSample = namedtuple('TextSample', ['token_ids', 'text'])

def log_score(model, tokenizer, samples, context="", grad=False, sum=True):
    # device = model.device
    if not model.config.is_encoder_decoder:
        context = tokenizer.bos_token if "" == context else context
    tokenized_context = tokenizer([context] * len(samples), return_tensors="pt", add_special_tokens=True)
    tokenized_context["input_ids"] = tokenized_context["input_ids"]#.to(device)
    tokenized_context["attention_mask"] = tokenized_context["attention_mask"]#.to(device)

    tokenized_samples = dict()
    tokenized_samples["input_ids"] = torch.stack([sample.token_ids for sample in samples])#.to(device)

    first_eos_indices = get_token_first_indices(
            tokenized_samples["input_ids"],
            tokenizer.eos_token_id
        )
    tokenized_samples["attention_mask"] = torch.where(
            tokenizer.pad_token_id == tokenized_samples["input_ids"],
            0, 1
        )
    for i, ix in enumerate(first_eos_indices):
        if None != model.config.forced_bos_token_id and\
            model.config.forced_bos_token_id == tokenized_samples["input_ids"][i][0]:
                tokenized_samples["attention_mask"][i][0] = 0
        else:
            tokenized_samples["attention_mask"][i][0] = 1  # at least score one token
        if ix != -1:  # if there is an eos token
            tokenized_samples["attention_mask"][i][ix] = 1  # score first eos token
            tokenized_samples["attention_mask"][i][ix + 1:] = 0  # ignore everything after it
    tokenized_samples["attention_mask"] = tokenized_samples["attention_mask"]#.to(device)

    if model.config.is_encoder_decoder:
        shift = None
        last = None
        inputs = tokenized_context
        labels = tokenized_samples["input_ids"]
    else:
        shift = tokenized_context["input_ids"].shape[-1] - 1
        last = -1
        inputs = {
            "input_ids": torch.cat((tokenized_context["input_ids"], tokenized_samples["input_ids"]), 1),
            "attention_mask": torch.cat((tokenized_context["attention_mask"], tokenized_samples["attention_mask"]), 1)
        }
        labels = inputs["input_ids"]

    if grad:
        outputs = model(**inputs, labels=labels)
    else:
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)

    all_logprobs = outputs.logits[:, shift:last, :].log_softmax(-1) # [n_samples, length, vocab]
    seq_logprobs = torch.gather(
            all_logprobs, 2, tokenized_samples["input_ids"][:, :, None]
        ).squeeze(-1) # [n_samples, length]

    seq_logprobs = torch.where(1 == tokenized_samples["attention_mask"], seq_logprobs, torch.tensor(0.))#.to(device)

    return seq_logprobs.sum(dim=1) if sum else seq_logprobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str, 
                        choices=['google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl'],
                        default = 'google/flan-t5-large')
    parser.add_argument('--dataset_type', dest='dataset_type', type=str, default='dongyoung4091/hh-generated_flan_t5_large')
    parser.add_argument('--fbs', dest='fbs', type=int, default=64)
    args = parser.parse_args()
    pprint(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model.to(device)

    # Load dataset
    dataset = load_dataset(args.dataset_type)

    # Caluate a(x)
    logprobs_list = []
    df = pd.DataFrame(dataset['train'])
    for idx,row in tqdm(df.iterrows()):
        context = row['prompt']
        fbs_list = []
        for fbs_i in range(len(row['response'])//args.fbs):
            samples_fb = row['response'][fbs_i*args.fbs:(fbs_i+1)*args.fbs]
            samples_tokens = tokenizer(samples_fb,return_tensors="pt",padding=True)['input_ids']
            samples = [TextSample(ots, tokenizer.decode(ots)) for ots in samples_tokens]
            logprobs = log_score(model, tokenizer, samples, context, grad=False).tolist()
            fbs_list.append(logprobs)
        fbs_list = sum(fbs_list,[])
        logprobs_list.append(fbs_list)
        
    # Save
    save_model_name = args.model_name.split('/')[-1].replace('-','_')
    save_fname = f'hh-generated_{save_model_name}.csv'
    df['log_probs_'+model_name] = logprobs_list
    df.to_csv(save_fname,index=False)
    print(f'Saved to {save_fname}')
    
if __name__ == '__main__':
    main()