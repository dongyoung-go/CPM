# Compositional preference models for aligning LMs

We introduce [Compositional Preference Models (CPMs)](https://arxiv.org/abs/2310.13011v1), a novel framework for training robust and interpretable preference models.

The generic handling of language models and the generation depend on HuggingFace's Transformers library.

## Usage

- ```run.sh```: Run overall experiments
- ```feature_extract/annotation.sh```: Extract feature values using LM
- ```mle-train/logistic_fits.sh```: Train logistic classifier that combines feature values into single model
- ```reward_model/pm_training.sh```: Train standard preference model
- ```mle-train/preference_evaluation.sh```: Evaluate preference alignment with LLM

## Citing
```
@inproceedings{go2023compositional,
  title={Compositional Preference Models for Aligning LMs},
  author={Go, Dongyoung and Korbak, Tomasz and Kruszewski, Germ{\'a}n and Rozen, Jos and Dymetman, Marc},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
