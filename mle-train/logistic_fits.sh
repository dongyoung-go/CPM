subdatas=("train","test")
model_sizes=("large","base","small")

# main fit
for subdata in "${subdatas[@]}"; do
    # CPM-GPT3.5
    python fit_logistic.py --dataset_type dongyoung4091/shp_with_features_20k_flan_t5_large --g_fn --split ${subdata}
    # CPM-Flan-T5
    python fit_logistic.py --dataset_type dongyoung4091/shp_with_features_20k_flan_t5_large --g_fn --split ${subdata} --zeroshot_feature
    # CPM-GPT3.5
    python fit_logistic.py --dataset_type dongyoung4091/hh-rlhf_with_features_flan_t5_large --g_fn --split ${subdata}
    # CPM-Flan-T5
    python fit_logistic.py --dataset_type dongyoung4091/hh-rlhf_with_features_flan_t5_large --g_fn --split ${subdata} --zeroshot_feature
done

## datasize ablation
for subdata in "${subdatas[@]}"; do
    for data_size in 100 500 1000; do
        python fit_logistic.py --dataset_type dongyoung4091/hh-rlhf_with_features_flan_t5_large --split ${subdata} --zeroshot_feature --g_fn --subset ${data_size}
        python fit_logistic.py --dataset_type dongyoung4091/shp_with_features_20k_flan_t5_large --split ${subdata} --g_fn --subset ${data_size}
    done
done

## extractor size ablation
for subdata in "${subdatas[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        python fit_logistic.py --dataset_type dongyoung4091/hh-rlhf_with_features_flan_t5_large --split ${subdata} --g_fn --zeroshot_feature --zeroshot_LM_type ${model_size}
    done
done

## overoptimization plot
for subdata in "${subdatas[@]}"; do
    python fit_logistic.py --dataset_type dongyoung4091/hh-rlhf_with_features_flan_t5_large --split ${subdata} --g_fn --external_pm_pseudo_label
    python fit_logistic.py --dataset_type dongyoung4091/shp_with_features_20k_flan_t5_large --split ${subdata} --g_fn --external_pm_pseudo_label2

    python fit_logistic.py --dataset_type dongyoung4091/hh-rlhf_with_features_flan_t5_large --split ${subdata} --zeroshot_feature --g_fn --external_pm_pseudo_label
    python fit_logistic.py --dataset_type dongyoung4091/hh-rlhf_with_features_flan_t5_large --split ${subdata}test --zeroshot_feature --g_fn --external_pm_pseudo_label
done