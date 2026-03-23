# Keeping Base Model Distributions

**Research Question**: Does alignment (instruction tuning) destroy base model stylistic distributions, and can distribution interpolation recover them?

## Key Findings

- **Alignment IMPROVES stylistic control** (7.26/10 vs 3.74/10 for base model, p < 0.0001) — the aligned model is better at writing in specific styles because it can follow instructions
- **Base model distributions survive alignment intact** — logit-space interpolation between base and aligned models produces smooth, continuous style modulation
- **The base model can't differentiate styles** (ANOVA p=0.54 ns) while the aligned model can (p=0.002) — alignment adds an "addressing mechanism" to existing distributions
- **Ornate styles benefit most from alignment** — Poe (+3.5), noir (+4.3), Shakespeare (+3.8) show the largest alignment benefit; Hemingway (+1.7) shows the smallest
- **Distribution arithmetic works as a continuous dial** — α=0.75 achieves style fidelity statistically indistinguishable from the fully aligned model (p=0.051)

## Methodology

- **Models**: Qwen2.5-3B (base) and Qwen2.5-3B-Instruct (aligned)
- **Styles**: 7 target styles (Hemingway, Shakespeare, Poe, Austen, Tolkien, academic, noir)
- **Distribution Arithmetic**: Token-level logit interpolation at α ∈ {0, 0.25, 0.5, 0.75, 1.0}
- **Evaluation**: GPT-4.1 as style judge (210 evaluations) + automated diversity metrics
- **Hardware**: 2× NVIDIA RTX A6000, total runtime ~42 minutes

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv add transformers accelerate openai numpy scipy matplotlib seaborn

# Run
CUDA_VISIBLE_DEVICES=0,1 python src/run_experiment.py
python src/additional_analysis.py
```

Requires: OPENAI_API_KEY environment variable, 2 GPUs with ≥12GB each.

## File Structure

```
├── REPORT.md              # Full research report with analysis
├── README.md              # This file
├── planning.md            # Research plan
├── literature_review.md   # Pre-gathered literature review
├── resources.md           # Catalog of resources
├── src/
│   ├── config.py          # Experiment configuration
│   ├── generate.py        # Text generation (base, aligned, interpolated)
│   ├── evaluate.py        # GPT-4.1 style evaluation + automated metrics
│   ├── analyze.py         # Statistical analysis and visualization
│   ├── additional_analysis.py  # Inter-style discrimination analysis
│   └── run_experiment.py  # Main pipeline runner
├── results/
│   ├── config.json        # Experiment configuration snapshot
│   ├── generations/       # Raw generated text
│   ├── evaluations/       # Metrics and statistical tests
│   └── plots/             # Visualizations
├── papers/                # Downloaded research papers (24)
├── datasets/              # Downloaded datasets
└── code/                  # Cloned reference implementations
```

See [REPORT.md](REPORT.md) for full details.
