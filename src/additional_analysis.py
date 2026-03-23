"""Additional analysis: inter-style discrimination and unsolicited diversity."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from config import STYLES, EVALUATIONS_DIR, PLOTS_DIR


def load_data():
    with open("results/evaluations/all_metrics.json") as f:
        return json.load(f)

    with open("results/generations/all_generations.json") as f:
        return json.load(f)


def compute_inter_style_variance(metrics):
    """
    Compute how differentiated outputs are across styles for each condition.
    Higher inter-style variance = model is producing MORE distinct outputs per style.
    This measures distributional diversity — whether the model can access different modes.
    """
    conditions = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]

    results = {}
    for cond in conditions:
        style_means = {}
        for style in metrics:
            scores = []
            for topic in metrics[style]:
                if cond in metrics[style][topic]:
                    m = metrics[style][topic][cond]
                    if "style_fidelity_mean" in m:
                        scores.append(m["style_fidelity_mean"])
            if scores:
                style_means[style] = np.mean(scores)

        if len(style_means) >= 2:
            values = list(style_means.values())
            results[cond] = {
                "mean_style_fidelity": np.mean(values),
                "std_across_styles": np.std(values),
                "range": max(values) - min(values),
                "per_style": style_means,
            }

    return results


def compute_style_discrimination(metrics):
    """
    For each condition, compute how well we can discriminate which style was targeted.
    Use the spread of style fidelity scores: if all styles get similar scores,
    the model is being generic (not actually differentiating styles).
    """
    conditions = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]

    # For each condition, get all style fidelity scores by style
    results = {}
    for cond in conditions:
        style_scores = defaultdict(list)
        for style in metrics:
            for topic in metrics[style]:
                if cond in metrics[style][topic]:
                    m = metrics[style][topic][cond]
                    if "style_fidelity_mean" in m:
                        style_scores[style].append(m["style_fidelity_mean"])

        if len(style_scores) >= 2:
            # Compute F-statistic (one-way ANOVA) across styles
            groups = [v for v in style_scores.values() if len(v) >= 2]
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                results[cond] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_val),
                    "per_style_means": {s: float(np.mean(v)) for s, v in style_scores.items()},
                }

    return results


def plot_style_differentiation(metrics):
    """Plot how well each condition differentiates between styles."""
    conditions = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]
    cond_labels = ["Base", "α=0.25", "α=0.5", "α=0.75", "Aligned"]
    styles = sorted(metrics.keys())

    fig, axes = plt.subplots(1, len(conditions), figsize=(20, 5), sharey=True)

    for ax, cond, label in zip(axes, conditions, cond_labels):
        style_vals = {}
        for style in styles:
            scores = []
            for topic in metrics[style]:
                if cond in metrics[style][topic]:
                    m = metrics[style][topic][cond]
                    if "style_fidelity_mean" in m:
                        scores.append(m["style_fidelity_mean"])
            if scores:
                style_vals[style] = scores

        if style_vals:
            positions = range(len(style_vals))
            means = [np.mean(v) for v in style_vals.values()]
            stds = [np.std(v) for v in style_vals.values()]
            ax.barh(list(range(len(style_vals))), means, xerr=stds,
                    capsize=3, color='steelblue', alpha=0.7)
            ax.set_yticks(list(range(len(style_vals))))
            ax.set_yticklabels(list(style_vals.keys()), fontsize=8)
            ax.set_xlabel("Style Fidelity")
            ax.set_title(label)
            ax.set_xlim(0, 10)
            ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle("Style Differentiation Across Conditions", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/style_differentiation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/style_differentiation.png")


def plot_recovery_by_alpha(metrics):
    """Plot style fidelity recovery curves per style."""
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = sorted(metrics.keys())
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    conditions = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]

    for style in styles:
        means = []
        for cond in conditions:
            scores = []
            for topic in metrics[style]:
                if cond in metrics[style][topic]:
                    m = metrics[style][topic][cond]
                    if "style_fidelity_mean" in m:
                        scores.append(m["style_fidelity_mean"])
            means.append(np.mean(scores) if scores else np.nan)

        ax.plot(alphas, means, 'o-', label=style, linewidth=2, markersize=6)

    ax.set_xlabel("Interpolation α (0=base, 1=aligned)")
    ax.set_ylabel("Style Fidelity (1-10)")
    ax.set_title("Style Fidelity Recovery Curves by Style")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alphas)
    ax.set_xticklabels(["Base\n(α=0)", "0.25", "0.5", "0.75", "Aligned\n(α=1)"])
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/per_style_recovery.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/per_style_recovery.png")


def main():
    print("=" * 60)
    print("ADDITIONAL ANALYSIS")
    print("=" * 60)

    metrics = load_data()

    # Inter-style variance
    print("\n--- INTER-STYLE VARIANCE ---")
    isv = compute_inter_style_variance(metrics)
    for cond, data in isv.items():
        print(f"  {cond}: mean_SF={data['mean_style_fidelity']:.2f}, "
              f"std_across_styles={data['std_across_styles']:.2f}, "
              f"range={data['range']:.1f}")

    # Style discrimination (ANOVA)
    print("\n--- STYLE DISCRIMINATION (ANOVA) ---")
    disc = compute_style_discrimination(metrics)
    for cond, data in disc.items():
        sig = "***" if data["p_value"] < 0.001 else "**" if data["p_value"] < 0.01 else "*" if data["p_value"] < 0.05 else "ns"
        print(f"  {cond}: F={data['f_statistic']:.2f}, p={data['p_value']:.4f} {sig}")

    # Plots
    plot_style_differentiation(metrics)
    plot_recovery_by_alpha(metrics)

    # Save additional analysis
    with open(f"{EVALUATIONS_DIR}/additional_analysis.json", "w") as f:
        json.dump({
            "inter_style_variance": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                                          for kk, vv in v.items() if kk != 'per_style'}
                                     for k, v in isv.items()},
            "style_discrimination": disc,
        }, f, indent=2)

    print("\nAdditional analysis complete!")


if __name__ == "__main__":
    main()
