# Research Plan: Keeping Base Model Distributions

## Motivation & Novelty Assessment

### Why This Research Matters
Alignment (RLHF/DPO) is known to reduce the diversity of language model outputs, particularly stylistic capabilities like impersonating authors or matching literary tones. This "alignment tax" limits the creative utility of aligned models. Understanding whether KL regularization during alignment can preserve these distributions — and whether they can be recovered post-alignment via interpolation — has direct implications for building models that are both safe AND creatively capable.

### Gap in Existing Work
Based on the literature review:
- Kirk et al. (2023) showed KL penalties alone don't recover diversity (counterintuitive)
- Lin et al. (2023) showed weight interpolation beats KL penalties for preserving capabilities
- Zhou et al. (2024) showed distribution arithmetic can invert alignment
- **No work has specifically tested whether stylistic impersonation capabilities (writing like specific authors/genres) can be recovered through distribution interpolation after alignment**
- **No work has mapped which stylistic distributions are easy vs. hard to recover**

### Our Novel Contribution
We empirically test whether:
1. Base model stylistic capabilities survive alignment and can be recovered via distribution interpolation
2. Different interpolation methods (weight-space vs. logit-space) differ in their ability to recover style
3. Some stylistic distributions are fundamentally harder to recover than others
4. The β parameter in DPO controls which styles are preserved

### Experiment Justification
- **Experiment 1 (Style Baseline)**: Needed to quantify what stylistic capabilities exist in base vs. aligned models
- **Experiment 2 (Distribution Arithmetic)**: Tests logit-space interpolation for style recovery — novel for style specifically
- **Experiment 3 (DPO with varying β)**: Tests whether KL strength during training controls style preservation
- **Experiment 4 (Weight Interpolation)**: Tests weight-space recovery and compares to logit-space

## Research Question
Can KL-constrained alignment preserve base model stylistic distributions, and can distribution interpolation (logit-space or weight-space) recover them post-alignment?

## Hypothesis Decomposition
H1: Aligned models show significantly reduced stylistic diversity compared to base models
H2: Distribution arithmetic (logit-space interpolation) can recover stylistic capabilities at intermediate α values
H3: Weight-space interpolation achieves similar recovery to logit-space arithmetic
H4: Some styles (more distinctive/frequent in pretraining data) are easier to recover than others
H5: Higher β in DPO preserves more stylistic capability, but may reduce alignment quality

## Proposed Methodology

### Approach
Use paired base/instruct models (Qwen2.5-3B and Qwen2.5-3B-Instruct) to:
1. Generate text in specific author/genre styles from both models
2. Apply distribution arithmetic at various interpolation coefficients
3. Measure style fidelity, diversity, and alignment quality
4. Optionally train DPO at different β values if time permits

### Model Choice
- **Qwen2.5-3B** (base) and **Qwen2.5-3B-Instruct** (aligned): Both fit on a single A6000 in fp16 (~6GB each), allowing simultaneous loading for distribution arithmetic. Modern model with strong base capabilities.

### Style Tasks
Generate text in the style of:
- **Distinctive authors**: Hemingway (terse), Shakespeare (archaic), Poe (gothic), Austen (formal), Tolkien (epic)
- **Genre styles**: Academic paper, children's story, film noir narration, legal document, poetry
- These span a range from highly distinctive (Shakespeare) to moderate (academic), testing H4.

### Experimental Steps
1. Load both models on GPU
2. Create 10 style prompts × 5 test topics = 50 generation tasks
3. Generate 5 samples per task from: base model, aligned model, and 5 interpolation levels
4. Evaluate with automated metrics + GPT-4.1 as style judge
5. Statistical analysis of style preservation across conditions

### Baselines
1. Base model (upper bound on style diversity)
2. Aligned model with standard prompting (lower bound)
3. Aligned model with style instruction (test if instruction alone recovers style)

### Evaluation Metrics
1. **Style fidelity**: GPT-4.1 judge scoring (0-10) how well output matches target style
2. **Lexical diversity**: Distinct n-grams (1,2,3-gram), type-token ratio
3. **Semantic diversity**: Sentence-BERT cosine similarity between samples
4. **Style distinctiveness**: Whether outputs for different styles are distinguishable (inter-style variance)
5. **Coherence/quality**: GPT-4.1 judge on text quality (to track alignment quality)

### Statistical Analysis Plan
- Paired t-tests or Wilcoxon signed-rank for base vs. aligned comparisons
- ANOVA across interpolation levels with post-hoc Tukey HSD
- Effect sizes (Cohen's d) for all comparisons
- 95% confidence intervals on all metrics
- Significance level α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1 confirmed: Aligned models show 20-40% reduction in style diversity
- H2 confirmed: Intermediate α (0.3-0.7) recovers style while maintaining some alignment
- H3 partially confirmed: Weight interpolation works but may differ in quality
- H4 confirmed: Distinctive styles (Shakespeare, Poe) are easier to recover than subtle ones
- Trade-off curve between style fidelity and alignment quality as α varies

## Timeline and Milestones
1. Environment + model loading: 15 min
2. Style prompt design + generation pipeline: 30 min
3. Run all generations (base, aligned, interpolated): 45 min
4. GPT-4.1 evaluation: 30 min
5. Statistical analysis + visualization: 30 min
6. Documentation: 30 min

## Potential Challenges
- Model loading may be slow — mitigate with fp16 and efficient batching
- GPT-4.1 evaluation costs — limit to ~500 calls (~$5-10)
- Style judgment subjectivity — use structured rubrics and multiple criteria
- Distribution arithmetic implementation — adapt from emulated-disalignment code

## Success Criteria
- Clear quantitative evidence for/against style recovery via interpolation
- Statistically significant differences between conditions
- Visualization of the alignment-style trade-off curve
- Identification of easy vs. hard-to-recover styles
