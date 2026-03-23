# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Keeping Base Model Distributions" — investigating how KL divergence terms during alignment preserve base model capabilities and whether distribution interpolation can recover lost diversity.

## Papers
Total papers downloaded: 24

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Direct Preference Optimization | Rafailov et al. | 2023 | papers/rafailov2023_dpo.pdf | Foundation DPO method |
| RL with KL penalties as Bayesian inference | Korbak et al. | 2022 | papers/korbak2022_kl_bayesian_inference.pdf | KL = variational inference |
| Beyond Reverse KL: f-DPO | Wang et al. | 2023 | papers/wang2023_f_dpo.pdf | f-divergence alternatives |
| f-divergence Minimization for alignment | Go et al. | 2023 | papers/go2023_f_divergence_alignment.pdf | f-DPG framework |
| Token-level DPO | Zeng et al. | 2024 | papers/zeng2024_token_dpo.pdf | Token-level KL constraints |
| Chi-Squared Preference Optimization | Huang et al. | 2024 | papers/huang2024_chi_squared.pdf | χ² alternative to KL |
| RLHF Generalisation and Diversity | Kirk et al. | 2023 | papers/kirk2023_rlhf_diversity.pdf | RLHF mode collapse evidence |
| Mitigating the Alignment Tax | Lin et al. | 2023 | papers/lin2023_alignment_tax.pdf | Model averaging for alignment tax |
| Creativity Has Left the Chat | Mohammadi | 2024 | papers/mohammadi2024_creativity_debiasing.pdf | Creativity loss from alignment |
| One fish, two fish (conceptual diversity) | Murthy et al. | 2024 | papers/murthy2024_conceptual_diversity.pdf | Conceptual diversity reduction |
| Language Models Resist Alignment | Ji et al. | 2024 | papers/ji2024_resist_alignment.pdf | Elasticity of alignment |
| The Unlocking Spell on Base LLMs | Lin et al. | 2023 | papers/lin2023_unlocking_base.pdf | Superficial alignment hypothesis |
| Distributional to Overton Pluralism | Lake et al. | 2024 | papers/lake2024_distributional_pluralism.pdf | Alignment doesn't destroy info |
| Self-Distillation for Distribution Gap | Yang et al. | 2024 | papers/yang2024_self_distillation.pdf | Self-distillation preserves capabilities |
| Online Merging Optimizers | Lu et al. | 2024 | papers/lu2024_online_merging.pdf | Online weight merging in RLHF |
| Adding Alignment Control to LMs | Zhu et al. | 2025 | papers/zhu2025_alignment_control.pdf | λ interpolation for alignment |
| Emulated Disalignment | Zhou et al. | 2024 | papers/zhou2024_emulated_disalignment.pdf | Distribution arithmetic attack |
| BoNBoN Alignment | Gui et al. | 2024 | papers/gui2024_bonbon.pdf | Best-of-N distribution analysis |
| Asymptotics of LM Alignment | Yang et al. | 2024 | papers/yang2024_asymptotics_alignment.pdf | Closed-form KL-constrained RL |
| Alignment as Distribution Learning | Yun et al. | 2025 | papers/yun2025_alignment_distribution.pdf | Distribution learning framework |
| Diverse Preference Optimization | Lanchantin et al. | 2025 | papers/lanchantin2025_diverse_pref_opt.pdf | Diversity-aware DPO |
| Creative Preference Optimization | Ismayilzada et al. | 2025 | papers/ismayilzada2025_creative_pref_opt.pdf | Creativity-preserving alignment |
| InstructGPT | Ouyang et al. | 2022 | papers/ouyang2022_instructgpt.pdf | Original RLHF method |
| Helpful and Harmless Assistant | Bai et al. | 2022 | papers/bai2022_helpful_harmless.pdf | Anthropic RLHF foundations |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| UltraFeedback Binarized | HuggingFace | 61K pairs | DPO preference training | datasets/ultrafeedback_binarized/ | Primary alignment dataset |
| Anthropic HH-RLHF | HuggingFace | 161K pairs | RLHF preference training | datasets/anthropic_hh_rlhf/ | Secondary preference dataset |
| TruthfulQA | HuggingFace | 817 questions | Alignment eval | datasets/truthfulqa/ | Alignment tax measurement |
| WritingPrompts (test) | HuggingFace | 15K examples | Diversity measurement | datasets/writingprompts_test/ | Creative diversity testing |

See datasets/README.md for download instructions and detailed descriptions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| rlfh-gen-div | github.com/facebookresearch/rlfh-gen-div | RLHF diversity metrics | code/rlfh-gen-div/ | EAD, Sent-BERT, NLI metrics |
| emulated-disalignment | github.com/ZHZisZZ/emulated-disalignment | Distribution arithmetic | code/emulated-disalignment/ | Base/aligned logit interpolation |
| trl | github.com/huggingface/trl | DPO/PPO training | code/trl/ | Standard RLHF toolkit |
| sdft | github.com/sail-sg/sdft | Self-distillation | code/sdft/ | Distribution gap preservation |
| investigating-alignment | github.com/thomlake/investigating-alignment | Alignment analysis | code/investigating-alignment/ | Coverage metrics, ICL alignment |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder with diligent mode for 4 search queries covering KL divergence in alignment, RLHF diversity loss, model interpolation, and style/creativity
- Cross-referenced papers found across multiple searches
- Prioritized papers with relevance score ≥ 3

### Selection Criteria
- Direct relevance to KL divergence and base model distribution preservation
- Empirical evidence of alignment tax or diversity loss
- Methods for interpolating between base and aligned distributions
- Recency (2022-2025) with foundational works (2022) included

### Challenges Encountered
- Many arXiv IDs from Semantic Scholar URLs did not match the actual papers — required lookup of correct IDs
- PDF chunker mixed up files when all chunks went to the same directory (fixed by using per-paper subdirectories)

### Gaps and Workarounds
- No existing dataset specifically designed for measuring stylistic impersonation capability loss; recommend using authorship attribution datasets (Victorian-Era, Guardian) or creating custom style prompts
- No single paper addresses all aspects of the hypothesis; the synthesis of ~7 key papers is needed

## Recommendations for Experiment Design

### Primary dataset(s)
**UltraFeedback Binarized** for DPO training, **WritingPrompts** for diversity measurement, **TruthfulQA** for alignment verification.

### Baseline methods
1. Standard DPO (β ∈ {0.01, 0.1, 0.5})
2. DPO with JS-divergence (f-DPO)
3. Post-hoc weight interpolation α ∈ {0.0, 0.2, 0.5, 0.8, 1.0}
4. Logit-space distribution arithmetic (emulated fine-tuning style)
5. Best-of-N (N=16) as diversity upper bound

### Evaluation metrics
1. Diversity: EAD distinct n-grams, Sent-BERT similarity
2. Style: Authorship classifier accuracy on style-prompted generations
3. Alignment quality: Reward model win rate
4. Alignment tax: MMLU, TruthfulQA

### Code to adapt/reuse
- **trl** for DPO training with configurable β
- **emulated-disalignment** for distribution arithmetic implementation
- **rlfh-gen-div** for diversity metric computation
- **investigating-alignment** for coverage metrics
