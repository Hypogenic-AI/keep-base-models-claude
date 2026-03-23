# Cloned Repositories

## 1. rlfh-gen-div (Kirk et al.)
- **URL**: https://github.com/facebookresearch/rlfh-gen-div
- **Purpose**: Official implementation of "Understanding the Effects of RLHF on LLM Generalisation and Diversity"
- **Location**: code/rlfh-gen-div/
- **Key files**: Training scripts for SFT/RLHF/BoN, diversity metric computation, evaluation pipelines
- **Notes**: Contains the EAD, Sent-BERT, and NLI diversity metrics used in the paper. Essential reference for diversity measurement implementation.

## 2. emulated-disalignment (Zhou et al.)
- **URL**: https://github.com/ZHZisZZ/emulated-disalignment
- **Purpose**: Implementation of emulated disalignment — distribution arithmetic between base and aligned models
- **Location**: code/emulated-disalignment/
- **Key files**: Inference code for log-linear combination of base/aligned model logits
- **Notes**: Core implementation of π_ED(y_t) ∝ π_base(y_t)^{α+1} / π_align(y_t)^α. Directly useful for implementing distribution interpolation experiments.

## 3. trl (HuggingFace)
- **URL**: https://github.com/huggingface/trl
- **Purpose**: Transformer Reinforcement Learning library — standard toolkit for DPO/PPO/RLHF training
- **Location**: code/trl/
- **Key files**: `trl/trainer/dpo_trainer.py`, `trl/trainer/ppo_trainer.py`
- **Notes**: The standard library for running alignment experiments. Supports DPO with configurable β, various loss functions, and reference model management.

## 4. sdft (Yang et al.)
- **URL**: https://github.com/sail-sg/sdft
- **Purpose**: Self-Distillation Fine-Tuning — bridges distribution gap during fine-tuning
- **Location**: code/sdft/
- **Key files**: Training scripts for self-distillation approach
- **Notes**: Alternative approach to preserving base distribution through self-generated data.

## 5. investigating-alignment (Lake et al.)
- **URL**: https://github.com/thomlake/investigating-alignment
- **Purpose**: Code for "From Distributional to Overton Pluralism" — measuring alignment's effect on diversity
- **Location**: code/investigating-alignment/
- **Key files**: Evaluation code, coverage metrics, ICL alignment experiments
- **Notes**: Contains implementations of coverage metrics (Cover-LEX, Cover-SEM) and ICL-based alignment recovery.
