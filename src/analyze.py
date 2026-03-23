"""Statistical analysis and visualization of style preservation experiment results."""

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
from config import STYLES, ALPHA_VALUES, EVALUATIONS_DIR, PLOTS_DIR


def load_metrics():
    with open(f"{EVALUATIONS_DIR}/all_metrics.json") as f:
        return json.load(f)


def aggregate_by_condition(metrics):
    """Aggregate metrics across all styles and topics for each condition."""
    condition_data = defaultdict(lambda: defaultdict(list))

    for style_name in metrics:
        for topic in metrics[style_name]:
            for condition, m in metrics[style_name][topic].items():
                for metric_name, value in m.items():
                    if isinstance(value, (int, float)) and value >= 0:
                        condition_data[condition][metric_name].append(value)

    # Compute means and stds
    summary = {}
    for condition in condition_data:
        summary[condition] = {}
        for metric_name, values in condition_data[condition].items():
            summary[condition][metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "n": len(values),
                "values": values,
            }
    return summary


def aggregate_by_style(metrics):
    """Aggregate metrics for each style across topics."""
    style_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for style_name in metrics:
        for topic in metrics[style_name]:
            for condition, m in metrics[style_name][topic].items():
                for metric_name, value in m.items():
                    if isinstance(value, (int, float)) and value >= 0:
                        style_data[style_name][condition][metric_name].append(value)

    return style_data


def run_statistical_tests(summary):
    """Run statistical tests comparing conditions."""
    results = {}

    # Test: base vs aligned on style fidelity
    if "base" in summary and "aligned" in summary:
        for metric in ["style_fidelity_mean", "distinct_2", "self_bleu_4"]:
            if metric in summary["base"] and metric in summary["aligned"]:
                base_vals = summary["base"][metric]["values"]
                aligned_vals = summary["aligned"][metric]["values"]
                n = min(len(base_vals), len(aligned_vals))
                if n >= 3:
                    stat, pval = stats.mannwhitneyu(base_vals[:n], aligned_vals[:n], alternative='two-sided')
                    # Effect size: rank-biserial correlation
                    effect = 1 - (2 * stat) / (n * n) if n > 0 else 0
                    results[f"base_vs_aligned_{metric}"] = {
                        "U_statistic": float(stat),
                        "p_value": float(pval),
                        "effect_size_r": float(effect),
                        "n_base": len(base_vals),
                        "n_aligned": len(aligned_vals),
                        "base_mean": float(np.mean(base_vals)),
                        "aligned_mean": float(np.mean(aligned_vals)),
                    }

    # Test: interpolated conditions vs aligned
    for alpha in [0.25, 0.5, 0.75]:
        interp_key = f"interp_{alpha}"
        if interp_key in summary and "aligned" in summary:
            for metric in ["style_fidelity_mean", "distinct_2"]:
                if metric in summary[interp_key] and metric in summary["aligned"]:
                    interp_vals = summary[interp_key][metric]["values"]
                    aligned_vals = summary["aligned"][metric]["values"]
                    n = min(len(interp_vals), len(aligned_vals))
                    if n >= 3:
                        stat, pval = stats.mannwhitneyu(interp_vals[:n], aligned_vals[:n], alternative='two-sided')
                        results[f"interp{alpha}_vs_aligned_{metric}"] = {
                            "U_statistic": float(stat),
                            "p_value": float(pval),
                            "n": n,
                            "interp_mean": float(np.mean(interp_vals)),
                            "aligned_mean": float(np.mean(aligned_vals)),
                        }

    return results


def plot_interpolation_curve(summary):
    """Plot style fidelity and diversity as a function of interpolation alpha."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions_ordered = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]
    x_labels = ["Base\n(α=0)", "α=0.25", "α=0.5", "α=0.75", "Aligned\n(α=1)"]
    x_pos = [0, 0.25, 0.5, 0.75, 1.0]

    for ax, metric, title, ylabel in [
        (axes[0], "style_fidelity_mean", "Style Fidelity vs. Interpolation", "Style Fidelity (1-10)"),
        (axes[1], "distinct_2", "Lexical Diversity vs. Interpolation", "Distinct Bigrams"),
        (axes[2], "text_quality_mean", "Text Quality vs. Interpolation", "Text Quality (1-10)"),
    ]:
        means = []
        stds = []
        valid_x = []
        for cond, x in zip(conditions_ordered, x_pos):
            if cond in summary and metric in summary[cond]:
                means.append(summary[cond][metric]["mean"])
                stds.append(summary[cond][metric]["std"])
                valid_x.append(x)

        if means:
            ax.errorbar(valid_x, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
            ax.set_xlabel("Interpolation α (0=base, 1=aligned)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/interpolation_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/interpolation_curve.png")


def plot_style_heatmap(style_data):
    """Heatmap of style fidelity across styles and conditions."""
    conditions = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]
    style_names = sorted(style_data.keys())

    matrix = np.full((len(style_names), len(conditions)), np.nan)
    for i, style in enumerate(style_names):
        for j, cond in enumerate(conditions):
            if cond in style_data[style] and "style_fidelity_mean" in style_data[style][cond]:
                vals = style_data[style][cond]["style_fidelity_mean"]
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=["Base", "α=0.25", "α=0.5", "α=0.75", "Aligned"],
                yticklabels=style_names, ax=ax, vmin=1, vmax=10)
    ax.set_title("Style Fidelity by Style × Interpolation Level")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Target Style")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/style_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/style_heatmap.png")


def plot_diversity_comparison(summary):
    """Bar chart comparing diversity metrics across conditions."""
    conditions = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]
    labels = ["Base", "α=0.25", "α=0.5", "α=0.75", "Aligned"]

    metrics_to_plot = ["distinct_1", "distinct_2", "distinct_3"]
    metric_labels = ["Distinct 1-gram", "Distinct 2-gram", "Distinct 3-gram"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))
    width = 0.25

    for i, (metric, mlabel) in enumerate(zip(metrics_to_plot, metric_labels)):
        means = []
        stds = []
        for cond in conditions:
            if cond in summary and metric in summary[cond]:
                means.append(summary[cond][metric]["mean"])
                stds.append(summary[cond][metric]["std"])
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=mlabel, capsize=3)

    ax.set_xlabel("Condition")
    ax.set_ylabel("Distinct N-gram Ratio")
    ax.set_title("Lexical Diversity Across Interpolation Conditions")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/diversity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/diversity_comparison.png")


def plot_tradeoff_curve(summary):
    """Plot alignment quality vs style fidelity trade-off."""
    conditions = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]
    labels = ["Base\n(α=0)", "α=0.25", "α=0.5", "α=0.75", "Aligned\n(α=1)"]

    fig, ax = plt.subplots(figsize=(8, 6))

    style_vals = []
    quality_vals = []
    valid_labels = []

    for cond, label in zip(conditions, labels):
        if cond in summary:
            sf = summary[cond].get("style_fidelity_mean", {}).get("mean")
            tq = summary[cond].get("text_quality_mean", {}).get("mean")
            if sf is not None and tq is not None:
                style_vals.append(sf)
                quality_vals.append(tq)
                valid_labels.append(label)

    if style_vals:
        ax.plot(style_vals, quality_vals, 'o-', markersize=10, linewidth=2)
        for x, y, label in zip(style_vals, quality_vals, valid_labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(10, 10), fontsize=9)

        ax.set_xlabel("Style Fidelity (1-10)")
        ax.set_ylabel("Text Quality (1-10)")
        ax.set_title("Style Fidelity vs. Text Quality Trade-off")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/tradeoff_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/tradeoff_curve.png")


def plot_style_recovery_difficulty(style_data):
    """Bar chart showing how much style is recovered at α=0.5 relative to base, per style."""
    styles = sorted(style_data.keys())
    recovery_ratios = []

    for style in styles:
        base_sf = np.mean(style_data[style].get("base", {}).get("style_fidelity_mean", [0]))
        aligned_sf = np.mean(style_data[style].get("aligned", {}).get("style_fidelity_mean", [0]))
        interp_sf = np.mean(style_data[style].get("interp_0.5", {}).get("style_fidelity_mean", [0]))

        if base_sf > 0 and base_sf != aligned_sf:
            recovery = (interp_sf - aligned_sf) / (base_sf - aligned_sf) if base_sf != aligned_sf else 0
            recovery_ratios.append(min(max(recovery, -0.5), 1.5))  # Clip for display
        else:
            recovery_ratios.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if r > 0.5 else 'orange' if r > 0 else 'red' for r in recovery_ratios]
    bars = ax.bar(styles, recovery_ratios, color=colors)
    ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='Full recovery')
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='No recovery')
    ax.set_xlabel("Style")
    ax.set_ylabel("Recovery Ratio at α=0.5\n(0=aligned, 1=base)")
    ax.set_title("Style Recovery Difficulty: Which Styles Can Be Recovered?")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/style_recovery_difficulty.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/style_recovery_difficulty.png")


def run_analysis():
    """Run full analysis pipeline."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STYLE PRESERVATION EXPERIMENT - ANALYSIS PHASE")
    print("=" * 60)

    metrics = load_metrics()
    summary = aggregate_by_condition(metrics)
    style_data = aggregate_by_style(metrics)

    # Print summary table
    print("\n--- AGGREGATE METRICS BY CONDITION ---")
    conditions = ["base", "interp_0.25", "interp_0.5", "interp_0.75", "aligned"]
    header = f"{'Condition':<15} {'StyleFid':>10} {'TextQual':>10} {'Dist-2':>10} {'SelfBLEU4':>10} {'AvgLen':>10}"
    print(header)
    print("-" * len(header))
    for cond in conditions:
        if cond in summary:
            sf = summary[cond].get("style_fidelity_mean", {}).get("mean", float('nan'))
            tq = summary[cond].get("text_quality_mean", {}).get("mean", float('nan'))
            d2 = summary[cond].get("distinct_2", {}).get("mean", float('nan'))
            sb = summary[cond].get("self_bleu_4", {}).get("mean", float('nan'))
            al = summary[cond].get("avg_length", {}).get("mean", float('nan'))
            print(f"{cond:<15} {sf:>10.2f} {tq:>10.2f} {d2:>10.3f} {sb:>10.3f} {al:>10.1f}")

    # Statistical tests
    print("\n--- STATISTICAL TESTS ---")
    test_results = run_statistical_tests(summary)
    for test_name, result in test_results.items():
        p = result["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {test_name}: p={p:.4f} {sig}")

    # Save test results
    with open(f"{EVALUATIONS_DIR}/statistical_tests.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Per-style analysis
    print("\n--- STYLE FIDELITY BY STYLE (Base → Aligned) ---")
    for style in sorted(style_data.keys()):
        base_sf = style_data[style].get("base", {}).get("style_fidelity_mean", [])
        aligned_sf = style_data[style].get("aligned", {}).get("style_fidelity_mean", [])
        interp_sf = style_data[style].get("interp_0.5", {}).get("style_fidelity_mean", [])

        base_mean = np.mean(base_sf) if base_sf else float('nan')
        aligned_mean = np.mean(aligned_sf) if aligned_sf else float('nan')
        interp_mean = np.mean(interp_sf) if interp_sf else float('nan')
        drop = base_mean - aligned_mean
        print(f"  {style:<15}: base={base_mean:.1f}, α=0.5={interp_mean:.1f}, aligned={aligned_mean:.1f} (drop={drop:.1f})")

    # Generate plots
    print("\n--- GENERATING PLOTS ---")
    plot_interpolation_curve(summary)
    plot_style_heatmap(style_data)
    plot_diversity_comparison(summary)
    plot_tradeoff_curve(summary)
    plot_style_recovery_difficulty(style_data)

    # Save full summary
    summary_serializable = {}
    for cond in summary:
        summary_serializable[cond] = {}
        for metric in summary[cond]:
            summary_serializable[cond][metric] = {
                "mean": summary[cond][metric]["mean"],
                "std": summary[cond][metric]["std"],
                "n": summary[cond][metric]["n"],
            }

    with open(f"{EVALUATIONS_DIR}/summary_stats.json", "w") as f:
        json.dump(summary_serializable, f, indent=2)

    print("\nAnalysis complete!")
    return summary, style_data, test_results


if __name__ == "__main__":
    run_analysis()
