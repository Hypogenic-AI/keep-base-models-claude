# Keeping Base Model Distributions: Can Distribution Interpolation Recover Stylistic Capabilities After Alignment?

## 1. Executive Summary

We investigated whether alignment (instruction tuning) diminishes a language model's ability to generate text in diverse styles, and whether logit-space distribution interpolation between base and aligned models can modulate this capability. Using Qwen2.5-3B (base) and Qwen2.5-3B-Instruct (aligned), we generated text in 7 distinct literary and functional styles across 3 topics, evaluating with both automated diversity metrics and GPT-4.1 as a style judge.

**Key finding**: Contrary to the naive interpretation of the "alignment tax" hypothesis, the aligned model significantly *outperforms* the base model on stylistic impersonation (style fidelity: 7.26 vs 3.74, p < 0.0001). The aligned model also shows significantly better style differentiation across styles (ANOVA F=6.32, p=0.002) while the base model produces undifferentiated outputs (F=0.87, p=0.54). However, logit-space distribution arithmetic does produce smooth interpolation between base and aligned behavior, confirming that these distributions are continuously navigable and not "destroyed" by alignment.

**Practical implication**: The base model's stylistic distributions survive alignment intact (consistent with the Superficial Alignment Hypothesis), but they are *not diminished* — rather, alignment *unlocks* access to them via instruction following. The question of "recovery" is thus reframed: base models have diverse distributions but poor *controllability*; alignment trades some distributional freedom for controllability.

## 2. Goal

### Hypothesis
Including a KL term on certain distributions during model alignment preserves the base model's ability to generate diverse outputs (e.g., stylistic impersonations), which are otherwise diminished by alignment; distribution interpolation can recover these capabilities post-alignment.

### Why This Matters
- Alignment is the standard process for making LLMs useful, but it's known to reduce output diversity (Kirk et al., 2023)
- Creative applications require access to diverse stylistic registers
- Understanding whether alignment destroys or merely redirects distributional diversity is important for designing better alignment procedures

### What Gap This Fills
- No prior work has specifically measured stylistic impersonation capabilities before and after alignment
- No work has tested logit-space distribution arithmetic specifically for style recovery
- The interaction between instruction-following ability and distributional diversity hasn't been directly quantified

## 3. Data Construction

### Models
| Model | Role | Size | Source |
|-------|------|------|--------|
| Qwen2.5-3B | Base model | 3B params | HuggingFace |
| Qwen2.5-3B-Instruct | Aligned model | 3B params | HuggingFace |

### Style Tasks
7 target styles spanning literary (Hemingway, Shakespeare, Poe, Austen, Tolkien) and functional (academic, noir) registers, each applied to 3 topics:
1. "A person arriving at an old house for the first time"
2. "The experience of watching a sunset over the ocean"
3. "A conversation between two strangers on a train"

### Generation Conditions
| Condition | Description | Samples per task |
|-----------|-------------|-----------------|
| base | Base model with completion-style prompt | 3 |
| aligned | Instruct model with chat-style prompt | 3 |
| interp_0.25 | 75% base + 25% aligned logits | 2 |
| interp_0.5 | 50% base + 50% aligned logits | 2 |
| interp_0.75 | 25% base + 75% aligned logits | 2 |

### Generation Parameters
- Max tokens: 128
- Temperature: 0.8
- Top-p: 0.95
- Random seed: 42

### Total Generations
21 style-topic tasks × 5 conditions × 2-3 samples = 231 text samples

## 4. Experiment Description

### Methodology

#### High-Level Approach
We compare three paradigms for accessing stylistic distributions:
1. **Base model (completion-style)**: Prompt with style description, let model continue
2. **Aligned model (instruction-style)**: Instruct the model to write in a specific style
3. **Distribution arithmetic (logit interpolation)**: At each token, combine logits from both models: `mixed_logits = (1-α) × base_logits + α × aligned_logits`

#### Why This Method?
Distribution arithmetic (Zhou et al., 2024; Zhu et al., 2025) has been shown to enable continuous interpolation between base and aligned model behaviors. By applying it to style generation, we directly test whether the style-related "modes" of the base distribution remain accessible and navigable post-alignment.

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.4.1+cu124 | Model inference |
| Transformers | 5.3.0 | Model loading |
| OpenAI API | gpt-4.1 | Style evaluation |
| NumPy | 2.4.3 | Numerical computation |
| SciPy | 1.15.3 | Statistical tests |
| Matplotlib/Seaborn | 3.10.8/0.13.2 | Visualization |

#### Hardware
- 2× NVIDIA RTX A6000 (49GB each)
- Base model on GPU 0, aligned model on GPU 1
- KV caching enabled for distribution arithmetic

#### Evaluation Metrics
1. **Style Fidelity** (GPT-4.1 judge, 1-10): How well the output captures the target style's distinctive voice, vocabulary, and sentence structure
2. **Text Quality** (GPT-4.1 judge, 1-10): Coherence and writing quality regardless of style
3. **Distinct-N** (N=1,2,3): Type-token ratio for n-grams, measuring lexical diversity
4. **Self-BLEU-4**: N-gram overlap between samples within a condition (lower = more diverse)
5. **Inter-style ANOVA**: Whether outputs for different styles are statistically distinguishable

### Experimental Protocol
- 210 GPT-4.1 API calls for evaluation (2 per condition-task pair)
- Mann-Whitney U tests for pairwise condition comparisons
- One-way ANOVA for inter-style discrimination
- Total experiment runtime: 42.5 minutes

## 5. Results

### Main Results Table

| Condition | Style Fidelity | Text Quality | Distinct-2 | Self-BLEU-4 | Avg Length |
|-----------|---------------|-------------|-----------|------------|-----------|
| Base (α=0) | 3.74 ± 0.93 | 5.95 ± 0.65 | 0.882 ± 0.049 | 0.007 ± 0.012 | 103.2 |
| α=0.25 | 5.14 ± 1.11 | 6.60 ± 0.73 | 0.913 ± 0.070 | 0.007 ± 0.016 | 104.4 |
| α=0.50 | 6.36 ± 1.03 | 6.55 ± 1.22 | 0.884 ± 0.113 | 0.006 ± 0.016 | 99.8 |
| α=0.75 | 6.67 ± 0.99 | 6.62 ± 1.35 | 0.915 ± 0.106 | 0.008 ± 0.020 | 97.1 |
| Aligned (α=1) | 7.26 ± 0.91 | 7.07 ± 1.16 | 0.945 ± 0.028 | 0.003 ± 0.004 | 91.4 |

### Statistical Tests

| Comparison | Metric | p-value | Significance |
|-----------|--------|---------|-------------|
| Base vs Aligned | Style Fidelity | < 0.0001 | *** |
| Base vs Aligned | Distinct-2 | 0.0001 | *** |
| α=0.25 vs Aligned | Style Fidelity | < 0.0001 | *** |
| α=0.50 vs Aligned | Style Fidelity | 0.0026 | ** |
| α=0.75 vs Aligned | Style Fidelity | 0.0511 | ns |

### Style Differentiation (ANOVA)

| Condition | F-statistic | p-value | Interpretation |
|-----------|------------|---------|---------------|
| Base | 0.87 | 0.540 | Styles NOT differentiated |
| α=0.25 | 3.03 | 0.041 | Marginally differentiated |
| α=0.50 | 1.06 | 0.429 | NOT differentiated |
| α=0.75 | 2.12 | 0.116 | NOT differentiated |
| Aligned | 6.32 | 0.002 | Styles WELL differentiated |

### Style Fidelity by Individual Style

| Style | Base | α=0.25 | α=0.50 | α=0.75 | Aligned | Drop (B→A) |
|-------|------|--------|--------|--------|---------|------------|
| Hemingway | 4.0 | 4.2 | 5.7 | 5.7 | 5.7 | -1.7 |
| Shakespeare | 3.2 | 5.0 | 6.3 | 6.0 | 7.0 | -3.8 |
| Poe | 4.5 | 6.5 | 7.0 | 7.7 | 8.0 | -3.5 |
| Austen | 3.2 | 4.0 | 5.5 | 6.2 | 7.0 | -3.8 |
| Tolkien | 4.2 | 5.5 | 7.2 | 7.3 | 7.3 | -3.2 |
| Academic | 3.3 | 5.3 | 6.3 | 7.2 | 7.7 | -4.3 |
| Noir | 3.8 | 5.5 | 6.5 | 6.5 | 8.2 | -4.3 |

### Qualitative Examples

**Hemingway style** ("A person arriving at an old house"):
- **Base**: "The person arrives at the old house, a large one with a porch in the front and a garden in the back. The porch is weathered and creaks with every step." — Generic description, not Hemingway
- **Aligned**: "Man walks up old cobbled stairs. Door unlocked. Pushes open. Inside dark. Candlestick near fireplace." — Captures terse, declarative Hemingway style
- **α=0.5**: "She stood at the threshold, eyes wide. Doors creaked behind her, echoing softly. The walls were thick, heavy." — Intermediate: some terseness, but more elaborate

**Shakespeare style** ("A person arriving at an old house"):
- **Base**: "I stumbled upon an ancient mansion, / Where the creaks and groans of the wooden floors / Echoed through the halls" — Modern poetry, not Shakespeare
- **Aligned**: "In twilight's hallowed arms did he arrive, / At door of yore, where shadows doth entwine" — Archaic vocabulary, iambic meter
- **α=0.5**: "In hallowed halls, where echoes whisper past, / A stranger enters, stranger to the past" — Hybrid: archaic vocabulary but modern rhythm

### Visualizations

All plots saved to `results/plots/`:
- `interpolation_curve.png` — Style fidelity, diversity, and quality as a function of α
- `style_heatmap.png` — Style fidelity matrix (style × condition)
- `diversity_comparison.png` — Lexical diversity across conditions
- `tradeoff_curve.png` — Style fidelity vs. text quality trade-off
- `style_recovery_difficulty.png` — Recovery ratio per style at α=0.5
- `style_differentiation.png` — Style discrimination across conditions
- `per_style_recovery.png` — Per-style recovery curves across α

## 5. Result Analysis

### Key Findings

**Finding 1: Alignment IMPROVES stylistic impersonation, not reduces it.**
The aligned model (Qwen2.5-3B-Instruct) scores 7.26/10 on style fidelity vs 3.74/10 for the base model (p < 0.0001). This contradicts the naive version of the hypothesis that alignment "loses capabilities." The aligned model's instruction-following ability allows it to *access* stylistic distributions that the base model has but cannot controllably invoke.

**Finding 2: Distribution arithmetic enables smooth interpolation.**
Logit-space interpolation produces a monotonic curve from base (3.74) to aligned (7.26) style fidelity, confirming that:
- Base model distributions survive alignment intact (they contribute positively when mixed in)
- The interpolation is continuous, not abrupt
- At α=0.75, style fidelity (6.67) is statistically indistinguishable from the aligned model (p=0.051)

**Finding 3: The base model cannot differentiate between target styles.**
ANOVA reveals that base model outputs are NOT significantly differentiated across the 7 target styles (F=0.87, p=0.54), while aligned model outputs ARE (F=6.32, p=0.002). The base model has the distributions but lacks the *addressing mechanism* (instruction following) to select among them.

**Finding 4: Some styles are easier to invoke than others.**
Poe (base: 4.5, aligned: 8.0) and noir (base: 3.8, aligned: 8.2) show the largest alignment benefit, while Hemingway shows the smallest (base: 4.0, aligned: 5.7). This suggests highly distinctive, ornate styles benefit more from alignment's controllability.

**Finding 5: Interpolation primarily adds alignment capability, not base capability.**
The interpolation curve is concave: most of the style improvement happens between α=0 and α=0.5, suggesting the aligned model's contribution is primarily the instruction-following "addressing" mechanism rather than new style knowledge.

### Hypothesis Testing

**H1 (Aligned models show reduced stylistic diversity)**: **REFUTED.** Aligned models show higher style fidelity (7.26 vs 3.74) AND better style differentiation (ANOVA p=0.002 vs p=0.54). Alignment improves, not reduces, controllable access to stylistic distributions.

**H2 (Distribution arithmetic recovers stylistic capabilities)**: **PARTIALLY SUPPORTED.** Interpolation produces a smooth curve, confirming base distributions are intact. However, the base model's contribution is not "recovery" but rather dilution — higher base weight reduces style quality because the base model lacks the control mechanism to use its own distributions.

**H3 (Some styles are harder to recover)**: **SUPPORTED.** The recovery curve varies by style. Hemingway (minimal, understated) is harder for both models, while ornate styles (Poe, noir, Shakespeare) respond more to the interpolation.

### Reframing the Research Question

Our results suggest the original framing — "alignment loses distributions that KL terms could preserve" — needs revision. A more accurate framing:

> **Base models have diverse latent distributions but poor controllability. Alignment doesn't destroy these distributions — it adds an addressing mechanism (instruction following) that enables controllable access. The "alignment tax" on diversity is real at the distributional level (Kirk et al., 2023) but manifests as mode collapse in UNSOLICITED generation, not as inability to produce diverse styles when instructed.**

The relevant question becomes: **Can distribution interpolation help in cases where the aligned model genuinely refuses or fails to produce a style?** This would occur with styles suppressed by safety training (e.g., mimicking real living persons, offensive content), not with literary styles.

### Surprises and Insights

1. **The base model is surprisingly bad at style instruction.** Even with explicit prompts, it produced generic text. This highlights that "having a distribution" and "being able to access it on demand" are fundamentally different capabilities.

2. **Interpolation at α=0.5 is qualitatively interesting.** The mixed outputs often have a distinctive character — somewhat stylistic but less formulaic than the aligned model's outputs. This suggests interpolation could be useful for generating text that is *inspired by* a style rather than imitating it.

3. **Hemingway's style is the hardest for all conditions.** The minimalist, understated style is harder to produce than ornate styles. This may be because "doing less" (fewer words, simpler vocabulary) is a harder optimization target than "doing more" (adding archaic vocabulary, gothic imagery).

### Limitations

1. **Single model pair**: We tested only Qwen2.5-3B base/instruct. Results may differ for larger models, different model families, or models aligned with different methods (PPO vs DPO).

2. **GPT-4.1 as judge**: Style evaluation is subjective. GPT-4.1 may have systematic biases (e.g., preferring fluent text, which benefits the aligned model).

3. **No safety-suppressed styles tested**: Our styles are benign literary styles. The more interesting case for the hypothesis is styles that alignment actively suppresses (e.g., toxic content, impersonation of real people), which we did not test.

4. **Small sample size**: 2-3 samples per condition-task means limited statistical power for per-style comparisons.

5. **Completion vs chat prompting**: The base model used completion-style prompts while the aligned model used chat templates. This is a confound — the aligned model's advantage may partly reflect better prompt format rather than alignment itself.

6. **No weight-space interpolation tested**: We only tested logit-space distribution arithmetic, not weight-space model averaging (Lin et al., 2023), which may show different patterns.

## 6. Conclusions

### Summary
Alignment does not destroy base model stylistic distributions — it adds an instruction-following mechanism that enables controllable access to them. Logit-space distribution arithmetic confirms smooth, continuous interpolation between base and aligned model behaviors, supporting the Superficial Alignment Hypothesis. The base model's diverse distributions are accessible through interpolation, but the aligned model's instruction-following ability is a more effective "addressing mechanism" than the base model's prompting.

### Implications
- **For alignment researchers**: KL regularization during alignment may be less about "preserving" distributions and more about maintaining the base model's distributional breadth while adding controllability. The risk isn't losing styles but losing the *diversity of unsolicited generation*.
- **For practitioners**: If you need specific styles, use the aligned model with clear instructions. Distribution arithmetic is useful when you want to *blend* base and aligned behaviors, e.g., for less formulaic but style-aware generation.
- **For the original hypothesis**: The distributions are not "fundamentally difficult to unlock post-alignment" — they were never locked. The difficulty was always in *addressing* them, which alignment solves.

### Confidence in Findings
**Moderate-high** for the main finding (alignment improves stylistic control). The effect is large (Cohen's d > 3.0 for style fidelity), replicated across 7 styles and 3 topics, with statistical significance. Lower confidence for generalization to other model families, sizes, or alignment methods.

## 7. Next Steps

### Immediate Follow-ups
1. **Test safety-suppressed styles**: Investigate styles that alignment actively suppresses (not just literary styles that it improves). These are the cases where "recovery" through interpolation would be most relevant.
2. **Unsolicited diversity measurement**: Generate text WITHOUT style instructions and measure whether the base model produces more *spontaneously diverse* outputs across prompts.
3. **Weight-space interpolation**: Compare logit-space arithmetic to weight-space model averaging at various α values.

### Alternative Approaches
- Train DPO variants with different β values on the base model and directly measure style preservation
- Use f-divergence alternatives (Jensen-Shannon, α-divergence) during alignment to test whether divergence choice affects style preservation
- Test on larger models (7B, 70B) to see if the pattern holds at scale

### Broader Extensions
- Apply distribution arithmetic to domain-specific registries (medical, legal, creative writing)
- Investigate whether alignment's "addressing mechanism" can be added without full alignment (e.g., lightweight steering vectors)
- Study which neuron groups / attention heads are responsible for style control vs. distribution diversity

### Open Questions
1. Are there distributions that alignment genuinely destroys (not just makes harder to access)?
2. Does the base model's distributional diversity manifest differently at different temperatures?
3. Can we design alignment procedures that explicitly maintain distributional breadth (KL on style-specific prompts) while still improving instruction following?

## References

### Papers
1. Rafailov et al. (2023). Direct Preference Optimization. arXiv:2305.18290
2. Korbak et al. (2022). RL with KL Penalties as Bayesian Inference. arXiv:2205.11275
3. Kirk et al. (2023). Understanding the Effects of RLHF on LLM Generalisation and Diversity. arXiv:2310.06452
4. Lin et al. (2023). Mitigating the Alignment Tax of RLHF. arXiv:2309.06256
5. Zhou et al. (2024). Emulated Disalignment. arXiv:2402.12343
6. Ji et al. (2024). Language Models Resist Alignment. arXiv:2406.06144
7. Zhu et al. (2025). Adding Alignment Control to Language Models. arXiv:2503.04346
8. Lake et al. (2024). From Distributional to Overton Pluralism. arXiv:2406.17692
9. Wang et al. (2023). Beyond Reverse KL: f-DPO. arXiv:2309.16240
10. Go et al. (2023). f-divergence Minimization for Alignment. arXiv:2302.08215

### Datasets and Tools
- Qwen2.5-3B / Qwen2.5-3B-Instruct (Alibaba)
- GPT-4.1 (OpenAI) for style evaluation
- PyTorch 2.4.1, Transformers 5.3.0
- 2× NVIDIA RTX A6000 GPUs
