from common import *
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import wandb
import dill as pickle
dataset_url = f'dongyoung4091/shp_with_features_20k_flan_t5_large'
output_path = f'out/{dataset_url.split("/")[1]}_coefficients.pkl'
np.seterr(all='raise')

def sanity_check(data_idx=[0],feature_names = ['specificity','intent', 'factuality', 'easy-to-understand', 'relevance', 'readability', 'enough-detail', 'biased:', 'fail-to-consider-individual-preferences', 'repetetive', 'fail-to-consider-context', 'too-long']):
    ds = load_dataset(dataset_url, split='train')
    
    df = pd.DataFrame(ds)
    df = df.iloc[data_idx,:].round()
    del df['__index_level_0__']
    ds_short = Dataset.from_pandas(df)
    
    options = ['A', 'B']

    feature_values, lm_log_scores, preferences = parse_ds(ds_short, feature_names, options, dict())

    preferences = preferences_as_str(preferences)

    target_moments = get_preferred_option_values(feature_values, preferences).mean(0) 

    print('target moments', target_moments)

    coefficients = train_coefficients(target_moments, lm_log_scores, feature_values, preferences)

    print('target_moments', target_moments)
    print('lm_log_scores', lm_log_scores)
    print('feature_values', feature_values)
    print('coefficients', coefficients)
    
def train_coefficients(target_moments, lm_log_scores, feature_values, preferences, tolerance=1e-3, lr=0.1):
    # wandb.init(project="lll")
    coefficients = np.zeros(len(target_moments))
    cnt = 0
    while True:
        ebm_scores = {option: 
                get_ebm_scores(lm_log_scores[option], feature_values[option], coefficients)
                for option in feature_values.keys()}

        ebm_moments = get_ebm_moments(ebm_scores, feature_values)

        grad = (target_moments - ebm_moments).astype(np.float32)

        coefficients += lr * grad

        err = np.abs(grad).max()
        lll = calc_lll(ebm_scores, preferences)
        acc = calc_acc(ebm_scores, preferences)

        messages = \
            [f'Err: {err:02.2g}'] + \
            [f'LLL: {lll:02.2g}'] + \
            [f'Acc: {acc:02.2g}']
        cnt += 1
        if cnt%1000==0:
            # wandb.log({"acc": acc, "lll": lll, "err": err})
            print({"acc": acc, "lll": lll, "err": err})
        if err < tolerance:
            for i in range(len(messages)):
                print()
            break
    print({"acc": acc, "lll": lll, "err": err})
    return coefficients

# https://www.wolframalpha.com/input?i=abs%28%28%28exp%28-546%29*exp%28x*10%29*10%2Bexp%28-206%29*exp%28x*9%29*9%29%2F%28exp%28-546%29*exp%28x*10%29%2Bexp%28-206%29*exp%28x*9%29%29%29-10%29%3C0.001
# answer: x>346.907
sanity_check(data_idx=[0],feature_names = ['specificity'])
# target_moments [10.]
# lm_log_scores {'A': array([-207.]), 'B': array([-546.])}
# feature_values {'A': array([[9.]]), 'B': array([[10.]])}
# coefficients [345.90689385]

# https://www.wolframalpha.com/input?i=abs%2818-%28%28%28exp%28-546%29*exp%28x*10%29*10%2Bexp%28-206%29*exp%28x*9%29*9%29%2F%28exp%28-546%29*exp%28x*10%29%2Bexp%28-206%29*exp%28x*9%29%29%29%2B%28%28exp%28-225%29*exp%28x*8%29*8%2Bexp%28-192%29*exp%28x*4%29*4%29%2F%28exp%28-225%29*exp%28x*8%29%2Bexp%28-192%29*exp%28x*4%29%29%29%29%29%3C0.002
# answer: x>346.213
sanity_check(data_idx=[0,1],feature_names = ['specificity'])
# target_moments [9.]
# lm_log_scores {'A': array([-207., -192.]), 'B': array([-546., -226.])}
# feature_values {'A': array([[9.],
#        [4.]]), 'B': array([[10.],
#        [ 8.]])}
# coefficients [345.21271789]

# https://www.wolframalpha.com/input?i=abs%28%28%28exp%28x*10%2By*8-546%29*10%2Bexp%28x*9%2By*5-207%29*9%29%2F%28exp%28x*10%2By*8-546%29%2Bexp%28x*9%2By*5-207%29%29%29-10%29%3C0.001%2C+abs%28%28%28exp%28x*10%2By*8-546%29*8%2Bexp%28x*9%2By*5-207%29*5%29%2F%28exp%28x*10%2By*8-546%29%2Bexp%28x*9%2By*5-207%29%29%29-8%29%3C0.001%2Cx+%3D+34.7
# answer: x\simeq 34.7, y>104.102
sanity_check(data_idx=[0],feature_names = ['specificity','too-long'])
# target_moments [10.  8.]
# lm_log_scores {'A': array([-207.]), 'B': array([-546.])}
# feature_values {'A': array([[9., 5.]]), 'B': array([[10.,  8.]])}
# coefficients [ 34.70063746 104.10191478]
