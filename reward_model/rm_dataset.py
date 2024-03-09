from transformers import T5Tokenizer
from torch.utils.data import DataLoader, Dataset
import torch

def preprocess_shp_dataset(example,prepend_mode=None):
    # For fitting RM, we don't need to unify 
    if 'hh-rlhf' in prepend_mode:
        prepend = '\n\nHuman: ' + example['history'] + '\n\nAssistant: '  
    elif 'shp' in prepend_mode:
        prepend = '\n\nPOST: ' + example['history'] + '\n\nResponse: '  
    else:
        prepend = ''
        
    original_A_col = [col for col in example.keys() if 'A' in col]
    original_B_col = [col for col in example.keys() if 'B' in col]
    
    if 'prepend_split' in prepend_mode:
        example['prepend'] = prepend
        ref_A = example['human_ref_A']
        ref_B = example['human_ref_B']
    else:
        ref_A = prepend + example['human_ref_A']
        ref_B = prepend + example['human_ref_B']
        
    if example['labels']==1:
        example['chosen'] = ref_A
        example['rejected'] = ref_B
        for col in original_A_col:
            example[col.replace('A','chosen')] = example[col]
        for col in original_B_col:
            example[col.replace('B','rejected')] = example[col]
    else:
        example['chosen'] = ref_B
        example['rejected'] = ref_A
        for col in original_B_col:
            example[col.replace('B','chosen')] = example[col]
        for col in original_A_col:
            example[col.replace('A','rejected')] = example[col]

    return example

def preprocess_hh_dataset(example,prepend_mode=None):
    bot_prefix = '\n\nAssistant: '
    prepend, response = example['chosen'].split(bot_prefix)
    prepend += bot_prefix 
        
    if 'prepend_split' in prepend_mode:
        example['prepend'] = prepend
        example['chosen'] = response
        example['rejected'] = example['rejected'].split(bot_prefix)[1]
 
    return example

class PairwiseDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.target_columns = None
        self.options = None
    
    def __len__(self):
        return len(self.data)
    
    def put_target_info(self,target_columns, options):
        self.target_columns = target_columns
        self.options = options
    
    def __getitem__(self, index):
        result = dict()
        item = self.data[index]
        input1 = item['chosen']
        input2 = item['rejected']
        if 'prepend' in item:
            input3 = item['prepend']
            prepend_ids = self.tokenizer.encode(input3, return_tensors='pt',padding="max_length", max_length = 512, truncation=True)
            result.update({'prepend_ids': prepend_ids,})
        else:
            prepend_ids = 0 # Won't be used
        
        chosen_ids = self.tokenizer.encode(input1, return_tensors='pt',padding="max_length", max_length = 512, truncation=True)
        rejected_ids = self.tokenizer.encode(input2, return_tensors='pt',padding="max_length", max_length = 512, truncation=True)
        result.update({'chosen_ids': chosen_ids,'rejected_ids': rejected_ids,})
        # Monkey patch for feature RM training to give chosen & rejected value
        if 'chosen_value' in item:
            result.update({'chosen_value':item['chosen_value'],'rejected_value':item['rejected_value']})
        if self.options is not None:
            if self.options == ['A','B']:
                if item['labels']==1:
                    item['chosen_value'] = torch.tensor([item[f'{col}_A'] for col in self.target_columns])
                    item['rejected_value'] = torch.tensor([item[f'{col}_B'] for col in self.target_columns])
                else:
                    item['chosen_value'] = torch.tensor([item[f'{col}_B'] for col in self.target_columns])
                    item['rejected_value'] = torch.tensor([item[f'{col}_A'] for col in self.target_columns])
            elif self.options == ['chosen','rejected']:
                item['chosen_value'] = torch.tensor([item[f'{col}_chosen'] for col in self.target_columns])
                item['rejected_value'] = torch.tensor([item[f'{col}_rejected'] for col in self.target_columns])
            result.update({'chosen_value':item['chosen_value'],'rejected_value':item['rejected_value']})
        
        return result