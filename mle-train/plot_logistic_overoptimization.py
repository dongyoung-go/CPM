import sys
sys.path.append('../feature_extract')
from reward_scaling_lll import *
import dill as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_corr_data(scored_samples, bon_df, save_name=None):
    plt.figure(figsize=(10,4),dpi=300)

    plt.clf()
    sns.set_theme('paper')
    all_scores_A = scored_samples['score_A']
    all_scores_B = scored_samples['score_B']
    ax = plt.subplot(1,2,1)
    plt.scatter(all_scores_A, all_scores_B)
    plt.title(f'Correlation: {corr(scored_samples)[0]:0.3f}')
    plt.xlabel('Score according to PM A')
    plt.ylabel('Score according to PM B')


    ax = plt.subplot(1,2,2)
    sns.lineplot(data=bon_df, x='n', y='best_of_A_score_A', ax=ax, legend='full', label='PM A (used for argmax)')
    sns.lineplot(data=bon_df, x='n', y='best_of_A_score_B', ax=ax, legend='full', label='PM B')
    # plt.xscale('log')
    ax.set_xlabel('n')
    ax.set_ylabel('Score')
    
    print('save_name',save_name)
    if save_name is not None:
        plt.savefig(save_name, dpi=300)
        print(f'Plot saved to {save_name}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored-samples', type=Path, default='dongyoung4091/hh-generated_flan_t5_large_with_features2')
    ap.add_argument('--gold_model', type=Path, default= "shp_with_features_20k_flan_t5_large_train_logistic.pkl")
    ap.add_argument('--proxy_model', type=Path, default= "shp_with_features_20k_flan_t5_large_test_logistic_pseudo_label.pkl")
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    
    scored_samples = load_scored_samples(args.scored_samples)

    model_A = pickle.load(open(args.gold_model, 'rb'))
    model_B = pickle.load(open(args.proxy_model, 'rb'))

    scored_samples = load_scored_samples(args.scored_samples)

    scored_samples = add_callback_features(scored_samples, model_A['feature_callbacks'])

    scored_samples = scored_samples.dropna()  # some features might not have been properly scored

    features_values_matrix = features_values(scored_samples, model_A['feature_names'])

    scored_samples = add_model_scores('score_A', scored_samples, model_A, features_values_matrix,standardize=True)

    scored_samples = add_model_scores('score_B', scored_samples, model_B, features_values_matrix,standardize=True)

    bon_df = best_of_n(scored_samples)

    plot_corr_data(scored_samples, bon_df,args.output)

main()