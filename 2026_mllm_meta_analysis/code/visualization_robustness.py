# Author: Piyush Sharma
"""
visualization_robustness.py
============================
Additional publication-quality figures for robustness analyses.

Generates:
  8. Leave-one-out sensitivity forest plot (Figure 8)
  9. Galbraith (radial) plot (Figure 9)
  10. Pareto efficiency frontier (Figure 10)
  11. Influence diagnostics plot (Figure 11)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict
import os


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ============================================================================
# Figure 8: Leave-One-Out Forest Plot
# ============================================================================

def plot_leave_one_out(loo_df: pd.DataFrame, full_pooled: float, save_path: str):
    """
    Forest plot showing the pooled estimate when each model is excluded.
    """
    n = len(loo_df)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35 + 2)))

    loo_sorted = loo_df.sort_values("abs_delta", ascending=True).reset_index(drop=True)

    for i, (_, row) in enumerate(loo_sorted.iterrows()):
        color = "#e74c3c" if row["abs_delta"] > 0.05 else "#3498db"

        ax.plot(
            [row["ci_lower"], row["ci_upper"]], [i, i],
            color=color, linewidth=1.5, zorder=2
        )
        ax.scatter(row["pooled_d"], i, s=50, color=color,
                  edgecolors="black", linewidth=0.5, zorder=3)

        ax.annotate(
            f"Δd = {row['delta_d']:+.4f}  (I²={row['I_sq']:.1f}%)",
            xy=(row["ci_upper"], i),
            fontsize=6.5, va="center",
            xytext=(5, 0), textcoords="offset points",
        )

    # Full pooled estimate line
    ax.axvline(x=full_pooled, color="#2c3e50", linestyle="--",
              linewidth=1.2, label=f"Full pooled d = {full_pooled:.4f}")

    labels = [row["excluded_model"] for _, row in loo_sorted.iterrows()]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Pooled Effect Size (d) with Model Excluded")
    ax.set_title(
        "Leave-One-Out Sensitivity Analysis\n"
        "Effect of Excluding Each Model on the Pooled Estimate",
        fontsize=11, fontweight="bold"
    )
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 9: Galbraith (Radial) Plot
# ============================================================================

def plot_galbraith(galbraith_df: pd.DataFrame, pooled_effect: float, save_path: str):
    """
    Galbraith (radial) plot: z-score vs. precision.
    Points far from the regression line indicate heterogeneity sources.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = np.where(galbraith_df["outlier"], "#e74c3c", "#3498db")

    ax.scatter(
        galbraith_df["precision"], galbraith_df["z_score"],
        s=60, c=colors, edgecolors="black", linewidth=0.5,
        alpha=0.8, zorder=3
    )

    # Regression line through origin with slope = pooled effect
    x_range = np.linspace(0, galbraith_df["precision"].max() * 1.1, 100)
    ax.plot(x_range, pooled_effect * x_range,
           color="#2c3e50", linewidth=1.5, linestyle="-",
           label=f"Expected (d = {pooled_effect:.3f})")

    # ±2 SD band (approximate confidence region)
    ax.plot(x_range, pooled_effect * x_range + 2,
           color="#95a5a6", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.plot(x_range, pooled_effect * x_range - 2,
           color="#95a5a6", linewidth=0.8, linestyle="--", alpha=0.7,
           label="±2 SD bounds")

    # Label outliers
    for _, row in galbraith_df[galbraith_df["outlier"]].iterrows():
        ax.annotate(
            row["model"], (row["precision"], row["z_score"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
            color="#e74c3c", fontweight="bold"
        )

    # Label non-outliers more subtly
    for _, row in galbraith_df[~galbraith_df["outlier"]].iterrows():
        ax.annotate(
            row["model"], (row["precision"], row["z_score"]),
            fontsize=5.5, ha="left", va="bottom",
            xytext=(3, 3), textcoords="offset points",
            alpha=0.6
        )

    n_outliers = galbraith_df["outlier"].sum()
    ax.set_xlabel("Precision (1 / SE)")
    ax.set_ylabel("Standardized Effect (d / SE)")
    ax.set_title(
        f"Galbraith (Radial) Plot for Heterogeneity Assessment\n"
        f"Outliers (|residual| > 2): {n_outliers} models",
        fontsize=11, fontweight="bold"
    )
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 10: Pareto Efficiency Frontier
# ============================================================================

def plot_pareto_frontier(pareto_data: Dict, save_path: str):
    """
    Scatter plot of model performance vs. computational cost,
    with the Pareto frontier highlighted.
    """
    df = pareto_data["data"].copy()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Colors: Pareto-optimal vs non-optimal
    pareto_colors = np.where(df["pareto_optimal"], "#e74c3c", "#3498db")
    pareto_sizes = np.where(df["pareto_optimal"], 120, 60)

    ax.scatter(
        df["est_tflops"], df["d"],
        s=pareto_sizes, c=pareto_colors,
        edgecolors="black", linewidth=0.5,
        alpha=0.8, zorder=3
    )

    # Draw Pareto frontier line
    pareto_pts = df[df["pareto_optimal"]].sort_values("est_tflops")
    if len(pareto_pts) > 1:
        ax.plot(
            pareto_pts["est_tflops"], pareto_pts["d"],
            color="#e74c3c", linewidth=2, linestyle="--",
            alpha=0.5, zorder=2, label="Pareto Frontier"
        )

    # Label all models
    for _, row in df.iterrows():
        fontweight = "bold" if row["pareto_optimal"] else "normal"
        color = "#c0392b" if row["pareto_optimal"] else "#2c3e50"
        ax.annotate(
            row["model"], (row["est_tflops"], row["d"]),
            fontsize=6.5, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
            fontweight=fontweight, color=color
        )

    ax.set_xscale("log")
    ax.set_xlabel("Estimated TFLOPs per Forward Pass (log scale)")
    ax.set_ylabel("Standardized Effect Size (d)")
    ax.set_title(
        f"Accuracy--Efficiency Pareto Frontier\n"
        f"{pareto_data['n_pareto']} Pareto-optimal models | "
        f"Best EEP: {pareto_data['best_eep']}",
        fontsize=11, fontweight="bold"
    )

    legend_elements = [
        mpatches.Patch(color="#e74c3c", label="Pareto-Optimal"),
        mpatches.Patch(color="#3498db", label="Dominated"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 11: Influence Diagnostics
# ============================================================================

def plot_influence_diagnostics(influence_df: pd.DataFrame, save_path: str):
    """
    Bar plot of DFBETAS showing each model's influence on the pooled estimate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    inf_sorted = influence_df.sort_values("dfbetas", ascending=True).reset_index(drop=True)
    n = len(inf_sorted)
    k = n

    # Panel A: DFBETAS
    ax = axes[0]
    colors = ["#e74c3c" if row["influential"] else "#3498db"
              for _, row in inf_sorted.iterrows()]

    ax.barh(range(n), inf_sorted["dfbetas"], color=colors, edgecolor="black",
           linewidth=0.3, height=0.7)

    # Threshold lines
    threshold = 2.0 / np.sqrt(k)
    ax.axvline(x=threshold, color="#e74c3c", linestyle="--",
              linewidth=0.8, alpha=0.7, label=f"±{threshold:.2f} threshold")
    ax.axvline(x=-threshold, color="#e74c3c", linestyle="--",
              linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(inf_sorted["model"], fontsize=7)
    ax.set_xlabel("DFBETAS")
    ax.set_title("(a) Influence on Pooled Estimate (DFBETAS)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    # Panel B: Standardized residuals vs hat values
    ax2 = axes[1]
    colors2 = ["#e74c3c" if row["influential"] else "#3498db"
               for _, row in influence_df.iterrows()]
    sizes = 50 + 200 * influence_df["cooks_distance"] / (influence_df["cooks_distance"].max() + 0.01)

    ax2.scatter(
        influence_df["hat_value"], influence_df["std_residual"],
        s=sizes, c=colors2, edgecolors="black", linewidth=0.5, alpha=0.8
    )

    for _, row in influence_df[influence_df["influential"]].iterrows():
        ax2.annotate(
            row["model"], (row["hat_value"], row["std_residual"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
            color="#e74c3c", fontweight="bold"
        )

    ax2.axhline(y=2, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(y=-2, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Hat Value (Leverage)")
    ax2.set_ylabel("Standardized Residual")
    ax2.set_title("(b) Residuals vs. Leverage\n(size ∝ Cook's distance)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Generate All Robustness Figures
# ============================================================================

def generate_robustness_figures(results: Dict, robustness: Dict, output_dir: str):
    """Generate all robustness analysis figures."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING ROBUSTNESS FIGURES")
    print("=" * 60)

    pooled = results["overall_meta_analysis"]["pooled_effect"]

    # Figure 8: Leave-one-out
    print("\n[8/11] Leave-one-out sensitivity plot...")
    plot_leave_one_out(
        robustness["leave_one_out"], pooled,
        os.path.join(output_dir, "fig8_leave_one_out.png")
    )

    # Figure 9: Galbraith plot
    print("[9/11] Galbraith (radial) plot...")
    plot_galbraith(
        robustness["galbraith"], pooled,
        os.path.join(output_dir, "fig9_galbraith_plot.png")
    )

    # Figure 10: Pareto frontier
    print("[10/11] Pareto efficiency frontier...")
    plot_pareto_frontier(
        robustness["pareto"],
        os.path.join(output_dir, "fig10_pareto_frontier.png")
    )

    # Figure 11: Influence diagnostics
    print("[11/11] Influence diagnostics...")
    plot_influence_diagnostics(
        robustness["influence"],
        os.path.join(output_dir, "fig11_influence_diagnostics.png")
    )

    print("\n✓ All 4 robustness figures generated successfully.")
