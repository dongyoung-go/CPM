# Generate dataset based on 'Anthropic/hh-rlhf', 'stanfordnlp/shp'
./gen_base_data.sh
# Extract feature values using LM
./feature_extract/annotation.sh
# Fit logistic regression based on feature values
./mle-train/logistic_fit.sh
# Fit Standard PM
./reward_model/pm_training.sh
# Evaluate preference alignment with LLM
./mle-train/preference_evaluation.sh