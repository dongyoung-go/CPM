from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
token_len = lambda response, prompt: (tokenizer(response, return_tensors="pt", add_special_tokens=True)['input_ids']).shape[1]

def parse_ds(ds, feature_names, context, response_prefix, score_prefix, options, feature_callbacks={}):
    feature_values = {option: np.zeros((len(ds), len(feature_names))) for option in options}
    lm_log_scores = {option: np.zeros(len(ds)) for option in options}
    preferences = []
    for i, row in enumerate(tqdm(ds, desc='parsing dataset')):
        for option in options:
            # extract dataset features
            for j, feature_name in enumerate(feature_names):
                key = f'{feature_name}_{option}'
                if key in row:
                    feature_values[option][i,j] = row[key]
                elif feature_name in feature_callbacks:
                    # use arbitrary functions to compute new features
                    feature_values[option][i,j] = feature_callbacks[feature_name](
                            row[f'{response_prefix}_{option}'], row[context])
                else:
                    raise RuntimeError(f"Unknown feature: {key}")
            lm_log_scores[option][i] = row[f'{score_prefix}_{option}']
        if isinstance(row['labels'], int):
            assert options == ['A','B'], "The dataset is assumed to be shp_with_features_20k_flan_t5_large. Please check your dataset"
            preferences.append(options[1 - row['labels']])
        else:
            preferences.append(row['labels'])
    return feature_values, lm_log_scores, preferences

def parse_features_ds(ds, feature_names, context_key, response_key, score_key, feature_callbacks={}):
    feature_values = np.zeros((len(ds), len(feature_names)))
    lm_log_scores = np.zeros(len(ds))
    index = {}
    for i, row in enumerate(tqdm(ds, desc='parsing dataset')):
        # extract dataset features
        for j, feature_name in enumerate(feature_names):
            key = feature_name
            if key in row:
                feature_values[i,j] = row[key]
            elif feature_name in feature_callbacks:
                # use arbitrary functions to compute new features
                feature_values[i,j] = feature_callbacks[feature_name](
                        row[response_key], row[context_key])
            else:
                raise RuntimeError(f"Unknown feature: {key}")
        lm_log_scores[i] = row[score_key]
        index[(row[context_key], row[response_key])] = i
    return feature_values, lm_log_scores, index

def preferences_as_str(preferences):
    return ['A' if x == 1 else 'B' for x in preferences]

def feature_values(ds, feature_names, option):
    values = np.zeros((len(ds), len(feature_names)))
    for i, row in enumerate(tqdm(ds, desc='loading feature values')):
        for j, feature_name in enumerate(feature_names):
            values[i,j] = row[f'{feature_name}_{option}']
    return values

def lm_log_scores(ds, option, score_prefix='log_score'):
    values = np.zeros(len(ds))
    for i, row in enumerate(tqdm(ds, desc='loading lm scores')):
        values[i] = row[f'{score_prefix}_{option}']
    return values

def get_ebm_scores(log_scores, all_feature_values, coefficients):
    return np.exp((log_scores + get_reward_scores(all_feature_values, coefficients)).astype(np.longdouble))

def get_reward_scores(all_feature_values, coefficients):
    return (coefficients * all_feature_values).sum(1)

def get_ebm_moments(ebm_scores, feature_values):
    z = sum(ebm_scores.values())
    non_zero_z = z > 0
    return np.mean(
            sum((ebm_scores[option][non_zero_z][...,np.newaxis] * feature_values[option][non_zero_z] for option in ebm_scores.keys())) /
            z[non_zero_z][...,np.newaxis], 0)

def calc_lll(ebm_scores, preferences, options):
    z = sum(ebm_scores.values())
    return (np.log(get_preferred_option_values(ebm_scores, preferences, options)) -
            np.log(z)).mean()

def calc_acc(ebm_scores, preferences, options):
    return np.mean(model_preferences(ebm_scores, options) == preferences)

def model_preferences(model_scores, options):
    return np.where(model_scores[options[0]] > model_scores[options[1]], options[0], options[1])

def get_preferred_option_values(values, preferences, options):
    preferred_option_values = np.zeros(values[options[0]].shape, dtype=values[options[0]].dtype)
    for i, preferred_option in enumerate(preferences):
        preferred_option_values[i] = values[preferred_option][i]
    return preferred_option_values

def standardize(feature_values):
    all_feature_values = np.concatenate(list(feature_values.values()), axis=0)
    mean = all_feature_values.mean(0)
    std = all_feature_values.std(0)
    print(std)
    return {option_name: (values - mean) / std
                for option_name, values in feature_values.items()}, mean, std
