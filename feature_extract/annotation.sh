# List of arguments for your Python script
subdatas=("train","test") # train, test are dataset with same size, which are used as two independent subset data for checking the model variance.
model_sizes=("xl","large","base","small")

# Loop through each argument
for subdata in "${subdatas[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        python annotate_new.py --dataset_name dongyoung4091/hh-rlhf_with_features_flan_t5_large --options chosen rejected  --context human --response-prefix assistant --split ${subdata} --model-name google/flan-t5-${model_size}
        python annotate_new.py --dataset_name dongyoung4091/hh-generated_flan_t5_large --eval-generated --split train --model-name google/flan-t5-${model_size}

        python annotate_new.py --dataset_name dongyoung4091/shp_with_features_20k_flan_t5_large --options A B  --context history --response-prefix human_ref --split ${subdata} --model-name google/flan-t5-${model_size}
        python annotate_new.py --dataset_name dongyoung4091/shp-generated_flan_t5_large --eval-generated --split train --model-name google/flan-t5-${model_size}
    done
done