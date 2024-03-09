# HH-RLHF evaluation
candidate1=out/hh-rlhf_with_features_flan_t5_large_flan_t5_zeroshot_train_logistic_zeroshot_g.pkl
candidate2=out/hh-rlhf_with_features_flan_t5_large_flan_t5_zeroshot_test_logistic_zeroshot_g.pkl
candidate3=../reward_model/hh-generated_flan_t5_large_annotated_google_flan-t5-xl_test.json
candidate3=../reward_model/hh-generated_flan_t5_large_annotated_google_flan-t5-xl_train.json
candidate4=dongyoung4091/hh-generated_flan_t5_rx_xl_all
python bon_candidate.py --scored-samples dongyoung4091/hh-generated_flan_t5_large_flan_t5_zeroshot --model-paths $candidate1 $candidate2 $candidate3 $candidate4 --chunk_eval
for i in {0..5}; do
    candidate1=out/hh-generated_flan_t5_large_flan_t5_zeroshot_${i}_vanila_score_vs_hh-rlhf_with_features_flan_t5_large_flan_t5_zeroshot_train_logistic_zeroshot_g.pkl.feather
    candidate2=out/hh-generated_flan_t5_large_flan_t5_zeroshot_${i}_vanila_score_vs_hh-rlhf_with_features_flan_t5_large_flan_t5_zeroshot_test_logistic_zeroshot_g.pkl.feather
    candidate3=out/hh-generated_flan_t5_large_flan_t5_zeroshot_${i}_vanila_score_vs_pm_score.feather
    candidate4=out/hh-generated_flan_t5_large_flan_t5_zeroshot_${i}_vanila_score_vs_external_rm1.feather
    python claude_eval.py --dataset $candidate1
    python claude_eval.py --dataset $candidate2
    python claude_eval.py --dataset $candidate3
    python claude_eval.py --dataset $candidate4
done

## scaling experiment
model_sizes=("small","base","large")
for model_size in "${model_sizes[@]}"; do
    candidate1=out/hh-rlhf_with_features_flan_t5_large_flan_t5_${model_size}_zeroshot_train_logistic_zeroshot_${model_size}_g_TEST_.pkl
    candidate2=out/hh-rlhf_with_features_flan_t5_large_flan_t5_${model_size}_zeroshot_test_logistic_zeroshot_${model_size}_g_TEST_.pkl
    python bon_candidate.py --scored-samples dongyoung4091/hh-generated_flan_t5_large_flan_t5_${model_size}_zeroshot --model-paths $candidate1 $candidate2 --chunk_eval
        for i in {0..5}; do
            candidate1=out/hh-generated_flan_t5_large_flan_t5_${model_size}_zeroshot_${i}_vanila_score_vs_hh-rlhf_with_features_flan_t5_large_flan_t5_${model_size}_zeroshot_train_logistic_zeroshot_${model_size}_g_TEST_.pkl.feather
            candidate2=out/hh-generated_flan_t5_large_flan_t5_${model_size}_zeroshot_${i}_vanila_score_vs_hh-rlhf_with_features_flan_t5_large_flan_t5_${model_size}_zeroshot_test_logistic_zeroshot_${model_size}_g_TEST_.pkl.feather
            python claude_eval.py --dataset $candidate1
            python claude_eval.py --dataset $candidate2
        done
done



# SHP evaluation
candidate1=out/shp_with_features_20k_flan_t5_large_flan_t5_zeroshot_train_logistic_zeroshot_g.pkl
candidate2=out/shp_with_features_20k_flan_t5_large_flan_t5_zeroshot_test_logistic_zeroshot_g.pkl
candidate3=../reward_model/shp-generated_flan_t5_large_annotated_google_flan-t5-xl_test.json
candidate3=../reward_model/shp-generated_flan_t5_large_annotated_google_flan-t5-xl_train.json
candidate4=data/shp-generated_flan_t5_large_flan_t5_zeroshot_sileod
python bon_candidate.py --scored-samples dongyoung4091/shp-generated_flan_t5_large_flan_t5_zeroshot --model-paths $candidate1 $candidate2 $candidate3 $candidate4 --chunk_eval
for i in {0..5}; do
    candidate1=out/shp-generated_flan_t5_large_flan_t5_zeroshot_${i}_vanila_score_vs_shp_with_features_20k_flan_t5_large_flan_t5_zeroshot_train_logistic_zeroshot_g.pkl.feather
    candidate2=out/shp-generated_flan_t5_large_flan_t5_zeroshot_${i}_vanila_score_vs_shp_with_features_20k_flan_t5_large_flan_t5_zeroshot_test_logistic_zeroshot_g.pkl.feather
    candidate3=out/shp-generated_flan_t5_large_flan_t5_zeroshot_${i}_vanila_score_vs_pm_score.feather
    candidate4=out/shp-generated_flan_t5_large_flan_t5_zeroshot_${i}_vanila_score_vs_external_rm1.feather_sileod
    python claude_eval.py --dataset $candidate1
    python claude_eval.py --dataset $candidate2
    python claude_eval.py --dataset $candidate3
    python claude_eval.py --dataset $candidate4
done
