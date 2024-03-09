for subdata in "${subdatas[@]}"; do
    # HH-RLHF dataset
    python RM_traing_from_scratch.py --model_name google/flan-t5-xl --save_model --micro_batch_size 4 --dataset_type dongyoung4091/hh-rlhf_with_features --subset ${subdata} --prepend_mode prepend_split --last_tkn --epochs 4
    # SHP dataset 
    python RM_traing_from_scratch.py --model_name google/flan-t5-xl --save_model --micro_batch_size 1 --dataset_type dongyoung4091/shp_with_features_20k --subset ${subdata} --prepend_mode shp-prepend_split --last_tkn --epochs 4 --lr 5e-5
done

# overoptimization
for subdata in "${subdatas[@]}"; do
    python RM_traing_from_scratch.py --model_name google/flan-t5-xl --save_model --micro_batch_size 1 --dataset_type dongyoung4091/hh-rlhf_with_features --subset ${subdata} --prepend_mode prepend_split --last_tkn --epochs 4 --pseudo_label
    python RM_traing_from_scratch.py --model_name google/flan-t5-xl --save_model --micro_batch_size 1 --dataset_type dongyoung4091/shp_with_features_20k --subset ${subdata} --prepend_mode shp-prepend_split --last_tkn --epochs 4 --pseudo_label --lr 5e-5
done

# subset_fit
for subdata in "${subdatas[@]}"; do
    for data_size in 100 500 1000; do
        python RM_traing_from_scratch.py --model_name google/flan-t5-xl --save_model --micro_batch_size 4 --dataset_type dongyoung4091/hh-rlhf_with_features --subset ${subdata} --prepend_mode prepend_split --last_tkn --epochs 4 --subset_fit ${data_size}
        python RM_traing_from_scratch.py --model_name google/flan-t5-xl --save_model --micro_batch_size 4 --dataset_type dongyoung4091/shp_with_features_20k --subset ${subdata} --prepend_mode shp-prepend_split --last_tkn --epochs 4 --subset_fit ${data_size}
    done
done

