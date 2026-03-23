# Literature Review: Keeping Base Model Distributions

## Research Area Overview

This review covers research on preserving base language model distributions during alignment (RLHF/DPO), the "alignment tax" on model capabilities, and techniques for interpolating between base and aligned distributions. The core question: can we align models to human preferences while retaining the base model's diverse generative capabilities (e.g., stylistic impersonation), and can we recover these capabilities post-alignment through distribution interpolation?

---

## Key Papers

### 1. Direct Preference Optimization (Rafailov et al., 2023)
- **arXiv:** 2305.18290 | **Citations:** 7694
- **Key Contribution:** Reformulates RLHF as a classification loss that implicitly optimizes a KL-constrained reward objective. The DPO loss is derived from the closed-form solution: π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β), where β controls KL divergence from the reference.
- **Relevance:** Foundation for understanding how KL regularization in alignment preserves base distributions. The β parameter directly controls the tradeoff between alignment strength and base distribution preservation.

### 2. RL with KL Penalties is Better Viewed as Bayesian Inference (Korbak et al., 2022)
- **arXiv:** 2205.11275 | **Citations:** 110
- **Key Contribution:** Shows KL-regularized RL is equivalent to variational inference: the base model is the prior, the reward provides the likelihood, and the aligned model is the posterior. Distribution collapse = posterior collapse.
- **Methodology:** Theoretical analysis showing the optimal KL-penalized policy has the form π*(x) ∝ π_ref(x) · exp(R(x)/β). Proposes alternatives beyond RL: conditional training, reward-weighted regression, and distributional policy gradients.
- **Relevance:** Provides the theoretical foundation for understanding β as a prior-likelihood tradeoff. Larger β preserves more of the base distribution (prior), smaller β allows more reward-driven drift.

### 3. Beyond Reverse KL: f-DPO (Wang et al., 2023)
- **arXiv:** 2309.16240 | **Citations:** 164
- **Key Contribution:** Generalizes DPO to arbitrary f-divergences (Jensen-Shannon, forward KL, α-divergences). Different divergences produce different alignment-diversity tradeoffs.
- **Results:** Jensen-Shannon divergence balances alignment and diversity better than reverse KL. The choice of divergence directly impacts generation diversity and calibration.
- **Relevance:** Shows that the *type* of distributional constraint matters, not just its strength. Alternative divergences may better preserve base model diversity.

### 4. Aligning LMs with Preferences through f-divergence Minimization (Go et al., 2023)
- **arXiv:** 2302.08215 | **Citations:** 117
- **Key Contribution:** Unifies RLHF and GDC frameworks under f-DPG. Shows no universally optimal divergence — different f-divergences present different alignment-diversity tradeoffs.
- **Results:** Jensen-Shannon divergence frequently outperforms forward KL by a wide margin. Divergence choice effects persist with model scale.
- **Relevance:** Empirically demonstrates that divergence selection is critical for preserving output diversity during alignment.

### 5. Understanding the Effects of RLHF on LLM Generalisation and Diversity (Kirk et al., 2023)
- **arXiv:** 2310.06452 | **Citations:** 304 | **Code:** github.com/facebookresearch/rlfh-gen-div
- **Key Contribution:** First rigorous empirical demonstration that RLHF causes both per-input and across-input mode collapse, while improving OOD generalization.
- **Methodology:** Three diversity metrics (EAD distinct n-grams, Sentence-BERT similarity, NLI diversity) measured on LLaMA-7B across summarization and instruction following. Compared SFT, RLHF (PPO), and Best-of-N.
- **Key Findings:**
  - RLHF substantially reduces per-input diversity vs SFT
  - RLHF reduces across-input diversity (mode collapse)
  - **Increasing the KL penalty does NOT recover diversity** — counterintuitively, higher KL penalties reduce diversity further while also hurting performance
  - Best-of-N preserves more diversity than RLHF, suggesting mode collapse is inherent to PPO optimization, not the reward model
- **Relevance:** Directly demonstrates the problem this research addresses. The finding that KL penalties alone cannot restore diversity motivates exploring alternative approaches (distribution interpolation, selective KL terms).

### 6. Mitigating the Alignment Tax of RLHF (Lin et al., 2023)
- **arXiv:** 2309.06256 | **Citations:** ~50
- **Key Contribution:** Proposes Model Averaging (MA) and Heterogeneous Model Averaging (HMA) as post-hoc methods to reduce alignment tax by interpolating between pre-RLHF and post-RLHF weights.
- **Methodology:** OpenLLaMA-3B and Mistral-7B, tested with RSF/DPO/PPO. Benchmarks: common sense QA, reading comprehension, translation, HH reward.
- **Key Findings:**
  - Simple weight interpolation θ_merged = (1-α)·θ_SFT + α·θ_RLHF dominates all other forgetting-mitigation methods (early stopping, L2, LoRA, KD, experience replay, KL penalty)
  - Averaging low-level (input) layers improves BOTH NLP tasks AND alignment reward simultaneously
  - HMA (different α per layer group) pushes the Pareto front beyond vanilla MA
  - α=0.2 is a robust default
  - KL penalty in PPO is "much less effective" than model averaging
- **Relevance:** Directly supports the hypothesis — weight-space interpolation between base and aligned models is the most effective way to preserve base capabilities. Layer-specific interpolation suggests certain distributions (lower layers) are more important to preserve.

### 7. Online Merging Optimizers (Lu et al., 2024)
- **arXiv:** 2405.17931 | **Citations:** ~20
- **Key Contribution:** Integrates model merging into every RLHF step (online) rather than post-hoc. Uses sparsification + consensus between gradient updates and SFT delta.
- **Methodology:** OnDARE (random sparsification) and OnTIES (magnitude + sign consensus). Tested on Qwen1.5-{1.8B, 7B} and LLaMA-3-8B with DPO/IPO/KTO.
- **Key Findings:**
  - Online merging consistently improves both preference scores AND benchmark performance, unlike offline merging which trades off between them
  - Complementary to KL constraints — works even with very low β
  - Training is stable even when discarding 99.95% of gradient modifications per step
- **Relevance:** Shows that continuous interpolation during training (not just post-hoc) better preserves base distributions while allowing reward optimization.

### 8. Language Models Resist Alignment (Ji et al., 2024)
- **arXiv:** 2406.06144 | **Citations:** 26
- **Key Contribution:** Demonstrates "elasticity" — aligned models tend to revert to the pretraining distribution upon further fine-tuning. Uses compression theory to explain why.
- **Key Findings:**
  - Alignment changes degrade k times faster than pretraining (k = |D_pretrain|/|D_align|, typically orders of magnitude)
  - Two-phase decline: rapid reversion toward pretraining distribution, then slow stabilization
  - Elasticity positively correlates with model size and pretraining data volume
  - Validated across SFT, PPO, DPO, KTO, SimPO
- **Relevance:** Strong evidence that base model distributions are inherently persistent and recoverable. The pretraining distribution acts as a powerful attractor — alignment is a thin, fragile overlay.

### 9. Emulated Disalignment (Zhou et al., 2024)
- **arXiv:** 2402.12343 | **Citations:** 36 | **Code:** github.com/ZHZisZZ/emulated-disalignment
- **Key Contribution:** Shows alignment can be reversed through simple distribution arithmetic: π_ED(y_t) ∝ π_base(y_t)^{α+1} / π_align(y_t)^α
- **Key Findings:**
  - Doubles the harmfulness of base models by inverting the alignment signal
  - Works across Llama-1, Llama-2, Mistral, Alpaca at 7B-70B scale
  - For strongly aligned models, less α is needed (alignment signal is larger)
  - The alignment delta is well-captured at the token level
- **Relevance:** Proves that base model distributions survive alignment fully intact. The alignment layer is a thin, arithmetically invertible transformation. Supports the hypothesis that distribution interpolation (positive α) can recover base model capabilities.

### 10. From Distributional to Overton Pluralism (Lake et al., 2024)
- **arXiv:** 2406.17692 | **Citations:** 32 | **Code:** github.com/thomlake/investigating-alignment
- **Key Contribution:** Shows alignment shifts diversity from distributional (diverse across samples) to Overton (diverse within a single longer response). Base model behavior is fully recoverable via in-context learning.
- **Key Findings:**
  - Little evidence that alignment suppresses useful information
  - Aligned model behavior can be recovered from base models using ICL + semantic hints
  - Supports the Superficial Alignment Hypothesis
- **Relevance:** Alignment doesn't destroy base distributions — it filters and aggregates them. The original distribution is accessible by prompting.

### 11. Adding Alignment Control to Language Models (Zhu et al., 2025)
- **arXiv:** 2503.04346 | **Citations:** 0
- **Key Contribution:** Adds a single identity-initialized layer before the transformer, trains only that layer with DPO. At inference, interpolates between aligned and unaligned paths using coefficient λ.
- **Key Findings:**
  - λ=1 matches full DPO with only ~2% parameters trained
  - Extrapolation (λ>1) surpasses full DPO (34.3% vs 26.2% on Arena-Hard)
  - Clear interpolation/extrapolation phenomena along the λ axis
  - Bottom layers matter far more than top layers for preference learning
- **Relevance:** Directly demonstrates that alignment can be treated as a separable, interpolatable transformation. The λ coefficient provides a continuous dial between base and aligned distributions.

### 12. Self-Distillation Bridges Distribution Gap (Yang et al., 2024)
- **arXiv:** 2402.13669 | **Citations:** 90 | **Code:** github.com/sail-sg/sdft
- **Key Contribution:** Uses model-generated data to bridge the distribution gap during fine-tuning, mitigating catastrophic forgetting of general capabilities.

---

## Common Methodologies

### KL-Constrained Optimization
The dominant framework: max E[R(y|x)] - β·KL(π||π_ref). Used in PPO-RLHF, DPO, and variants. β controls base distribution preservation. The closed-form optimum π* ∝ π_ref · exp(R/β) makes this equivalent to Bayesian posterior updating.

### f-Divergence Alternatives
Jensen-Shannon divergence consistently provides better alignment-diversity tradeoffs than reverse KL. Forward KL and α-divergences offer further options. The choice of divergence matters as much as its strength.

### Model Averaging / Interpolation
Post-hoc weight interpolation: θ_merged = (1-α)·θ_base + α·θ_aligned. Simple but highly effective. Layer-specific ratios (HMA) further improve results. Online merging during training is even better.

### Distribution Arithmetic
Log-linear combination of base and aligned model logits at inference time. Includes emulated fine-tuning, contrastive decoding, and proxy tuning. The formula π_EFT(y_t) ∝ π_base(y_t) · (π_aligned(y_t)/π_base(y_t))^α enables continuous interpolation.

---

## Standard Baselines
- **SFT** (supervised fine-tuning): Standard pre-RLHF baseline
- **DPO** with varying β: Primary alignment method
- **PPO** with KL penalty: Classic RLHF approach
- **Best-of-N** sampling: Inference-time alignment baseline (preserves more diversity)
- **Simple model averaging**: Post-hoc weight interpolation
- **LoRA**: Parameter-efficient alignment baseline

## Evaluation Metrics
- **Alignment quality:** MT-Bench, AlpacaEval (LC win rate), Arena-Hard, reward model scores
- **Diversity:** Distinct n-grams (EAD), Sentence-BERT cosine similarity, Self-BLEU, entropy
- **Alignment tax:** MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, HumanEval
- **Distribution distance:** KL divergence from reference policy, calibration (ECE)
- **Style preservation:** Authorship attribution accuracy, perplexity on style-specific corpora

## Datasets in the Literature
- **Preference data:** Anthropic HH-RLHF, UltraFeedback, TL;DR, AlpacaFarm
- **Diversity testing:** WritingPrompts, TL;DR (multi-sample generation)
- **Alignment tax benchmarks:** MMLU, HellaSwag, ARC, GSM8K, TruthfulQA

---

## Gaps and Opportunities

1. **Style-specific KL terms:** No paper has studied whether applying KL constraints *selectively* on certain distributions (e.g., style-related tokens/layers) better preserves stylistic capabilities than a uniform KL penalty.

2. **Post-alignment distribution interpolation for style recovery:** While weight interpolation and distribution arithmetic are well-studied for safety/capability, no work has specifically tested whether they can recover stylistic impersonation abilities lost during alignment.

3. **Identifying "fundamentally difficult to recover" distributions:** The elasticity paper shows most capabilities revert easily, but which ones don't? This gap directly relates to the hypothesis about distributions that are "fundamentally difficult to recover."

4. **Combining divergence choice with interpolation:** No work combines f-divergence selection (e.g., JS instead of KL) with post-hoc distribution interpolation. These may be complementary.

5. **Measuring creative/stylistic diversity specifically:** Current diversity metrics (distinct-n, Sent-BERT) are generic. No standardized benchmark exists for measuring stylistic impersonation capability before/after alignment.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **UltraFeedback Binarized** — for DPO alignment training (high quality, widely used)
2. **Anthropic HH-RLHF** — secondary preference dataset for validation
3. **WritingPrompts** — for measuring creative diversity before/after alignment
4. **TruthfulQA** — for measuring alignment gains
5. **MMLU** — for measuring alignment tax on knowledge

### Recommended Baselines
1. Standard DPO with varying β (0.01, 0.1, 0.5, 1.0)
2. DPO with JS-divergence instead of KL (f-DPO)
3. Post-hoc weight averaging at varying α (0.0 to 1.0)
4. Distribution arithmetic (emulated fine-tuning style interpolation)
5. Best-of-N sampling as an upper bound on diversity preservation

### Recommended Metrics
1. **Diversity:** EAD (distinct n-grams), Sentence-BERT cosine sim, per-input and across-input
2. **Style:** Author attribution accuracy using a trained classifier
3. **Alignment quality:** Reward model score, win rate vs reference
4. **Alignment tax:** MMLU, TruthfulQA accuracy
5. **Distribution distance:** KL divergence from base model

### Methodological Considerations
- Use a small model (1.5B-7B) to enable rapid iteration across many conditions
- Measure diversity at both the per-input and across-input levels (Kirk et al.)
- Test weight interpolation at multiple granularities (full model, per-layer, per-block)
- Compare weight-space interpolation vs. logit-space distribution arithmetic
- Include a "style impersonation" task (e.g., write in the style of X) to directly test the hypothesis about creative capability preservation
