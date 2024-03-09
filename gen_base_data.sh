# Generate dataset based on 'Anthropic/hh-rlhf', 'stanfordnlp/shp'
./feature_extract/sample_generate.py

# Add log_score (Optional)
train_data="dongyoung4091/hh-rlhf_with_features"
python compute-logprobs.py google/flan-t5-large --save_local --dataset $train_data --options chosen rejected --context human --response-prefix assistant
eval_data="dongyoung4091/hh-generated_flan_t5_large_with_features2"
python compute-logprobs.py google/flan-t5-large --load_local --save_local --dataset $eval_data --singleton --context prompt --response-prefix response

# Add log_score (Optional)
train_data="dongyoung/shp_with_features_20k"
python compute-logprobs.py google/flan-t5-large --save_local --dataset $train_data --options A B --context history --response-prefix human_ref
eval_data="dongyoung4091/shp-generated_flan_t5_large_with_features" 
python compute-logprobs.py google/flan-t5-large --save_local --dataset $eval_data --singleton --context prompt --response-prefix response
