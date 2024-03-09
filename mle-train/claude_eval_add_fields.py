import re
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description=
            """
            Evaluates the pairwise ranking accuracy of generated responses.
            """)
    ap.add_argument('--dataset', default='dongyoung4091/hh-generated_flan_t5_large')
    ap.add_argument('--split', default='train')
    ap.add_argument('--indices', nargs='+', type=int, default=[0, 1])
    args = ap.parse_args()

    assert len(args.indices)==2,' It should be pairwise comparision'
    identifier = '_'.join([str(idx) for idx in args.indices])
    dataset_name = args.dataset.split('/')[-1]
    save_fname = f'./data/{dataset_name}_claude_evaluation{identifier}.json'
    with open(save_fname) as f:
        data = json.load(f)

    dataset = load_dataset(args.dataset,split=args.split)

    data['original_response1'] = list(map(lambda responses: responses[args.indices[0]], dataset['response']))
    data['original_response2'] = list(map(lambda responses: responses[args.indices[1]], dataset['response']))
    data['original_prompt'] = list(dataset['prompt'])

    with open(save_fname, encoding="utf-8", mode="w") as f:
        json.dump(data, f)
    print(f"{save_fname} is modified and saved")
    
    directory_path = Path(save_fname).parent
    pattern = rf'{dataset_name}_claude_evaluation\d+_\d+\.json'
    save_concat_fname = directory_path /  f'{dataset_name}_claude_evaluation.json'
    # merge related jsons
    df = pd.concat([pd.read_json(directory_path / fname) for fname in os.listdir(directory_path) if re.match(pattern,fname)]).reset_index(drop=True)
    final_list = {col: df[col].tolist() for col in df.columns}
    with open(save_concat_fname, encoding="utf-8", mode="w") as file:
        json.dump(final_list, file)
    print(f"merged file is saved to {save_concat_fname}")

main()