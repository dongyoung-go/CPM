from disco.distributions.lm_distribution import LMDistribution, TextSample
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM
import argparse

def add_scores_singleton(example, lm, context, response_key, score_key):
    response = to_text_sample(example[response_key], lm.tokenizer)
    log_score = lm.log_score([response], context=example[context])
    example[score_key] = log_score.item()
    return example

def add_scores(example, lm, context, options, response_prefix, score_prefix):
    for option in options:
        option_response = to_text_sample(example[f'{response_prefix}_{option}'], lm.tokenizer)
        option_log_score = lm.log_score([option_response], context=example[context])
        example[f'{score_prefix}_{option}'] = option_log_score.item()
    return example

def to_text_sample(text, tokenizer):
    text += tokenizer.eos_token  # adding eos_token at the end
    token_ids = tokenize(text, tokenizer)
    return TextSample(text=text, token_ids=token_ids)

def tokenize(text, tokenizer):
    return tokenizer.encode(text, return_tensors='pt')[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_name', default='google/flan-t5-large')
    ap.add_argument('--singleton', action='store_true', default=False,
            help='there is a single sample per row (and thus, no options) in the ds')
    ap.add_argument('--dataset', default='dongyoung4091/shp_with_features_20k')
    ap.add_argument('--context', default='history')
    ap.add_argument('--options', nargs='+', default=['A', 'B'])
    ap.add_argument('--response-prefix', default='human_ref')
    ap.add_argument('--score-prefix', default='log_score')
    ap.add_argument('--save_local', action='store_true', help='Option to save data in local')
    ap.add_argument('--load_local', action='store_true', help='Option to load data from local')
    args = ap.parse_args()

    model_name = args.model_name

    lm = LMDistribution(model_name, auto=AutoModelForSeq2SeqLM, device='cuda')
    if args.load_local:
        ds = load_from_disk(args.dataset)
    else:
        ds = load_dataset(args.dataset)
    if args.singleton:
        ds = ds.map(add_scores_singleton, fn_kwargs={
            'lm': lm,
            'context': args.context,
            'response_key': args.response_prefix,
            'score_key': args.score_prefix})
    else:
        ds = ds.map(add_scores, fn_kwargs={
            'lm': lm,
            'context': args.context,
            'options': args.options,
            'response_prefix': args.response_prefix,
            'score_prefix': args.score_prefix})

    model_name_simple = model_name.split("/")[1].replace("-", "_")
    dataset_name = args.dataset.split('/')[1]
    target_dataset_name = f'{dataset_name}_{model_name_simple}'
    if args.save_local:
        fname = f'out/{target_dataset_name}'
        ds.save_to_disk(fname)
        print(f'Saved to {fname}')
    else:
        ds.push_to_hub(f'dongyoung4091/{target_dataset_name}')
    

main()
