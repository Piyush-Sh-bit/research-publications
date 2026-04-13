# Author: Piyush Sharma
"""
visualization.py
=================
Publication-quality figure generation for the MLLM meta-analysis.

Generates:
  1. Forest plot (Figure 1)
  2. Funnel plot (Figure 2)
  3. Benchmark correlation heatmap (Figure 3)
  4. Scale-performance scatter with regression (Figure 4)
  5. Subgroup analysis forest plot (Figure 5)
  6. Radar chart of top models (Figure 6)
  7. Box plot by training strategy (Figure 7)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from typing import Dict, Optional
import os

# ============================================================================
# Global plot configuration
# ============================================================================

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


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ============================================================================
# Figure 1: Forest Plot
# ============================================================================

def plot_forest(
    es_df: pd.DataFrame, 
    overall_ma: Dict, 
    save_path: str
):
    """
    Create a forest plot of model effect sizes with pooled estimate.
    
    Parameters
    ----------
    es_df : pd.DataFrame
        Effect size DataFrame (one row per model).
    overall_ma : dict
        Overall meta-analysis results.
    save_path : str
        Path to save the figure.
    """
    n = len(es_df)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4 + 2)))
    
    # Sort by effect size
    es_sorted = es_df.sort_values("d", ascending=True).reset_index(drop=True)
    
    y_positions = np.arange(n)
    
    # Color by training strategy
    strategy_colors = {
        "RLHF": "#e74c3c",
        "instruction_tuning": "#3498db",
        "pretraining_alignment": "#2ecc71",
    }
    
    for i, (_, row) in enumerate(es_sorted.iterrows()):
        color = strategy_colors.get(row["training_strategy"], "#95a5a6")
        
        # Confidence interval line
        ax.plot(
            [row["ci_lower"], row["ci_upper"]], 
            [i, i], 
            color=color, linewidth=1.5, zorder=2
        )
        
        # Effect size marker (size proportional to 1/variance)
        marker_size = max(30, min(200, 50 / row["var"]))
        ax.scatter(
            row["d"], i, 
            s=marker_size, color=color, 
            edgecolors="black", linewidth=0.5, zorder=3
        )
    
    # Pooled effect (diamond)
    pooled = overall_ma["pooled_effect"]
    pooled_ci = (overall_ma["pooled_ci_lower"], overall_ma["pooled_ci_upper"])
    diamond_y = -1.5
    diamond = plt.Polygon(
        [[pooled_ci[0], diamond_y], [pooled, diamond_y - 0.3],
         [pooled_ci[1], diamond_y], [pooled, diamond_y + 0.3]],
        facecolor="#2c3e50", edgecolor="black", linewidth=0.8, zorder=3
    )
    ax.add_patch(diamond)
    
    # Zero reference line
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    
    # Pooled estimate vertical line
    ax.axvline(x=pooled, color="#2c3e50", linestyle=":", linewidth=0.8, alpha=0.5)
    
    # Labels
    labels = []
    for _, row in es_sorted.iterrows():
        label = f"{row['model']}  ({row['params_b']:.0f}B)"
        labels.append(label)
    labels.append(f"\nPooled (k={overall_ma['k']})")
    
    all_y = list(y_positions) + [diamond_y]
    ax.set_yticks(all_y)
    ax.set_yticklabels(labels, fontsize=8)
    
    # Right-side annotations (effect size values)
    for i, (_, row) in enumerate(es_sorted.iterrows()):
        ax.annotate(
            f"{row['d']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]",
            xy=(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 2.5, i),
            fontsize=7, va="center",
            xytext=(5, 0), textcoords="offset points",
        )
    
    ax.set_xlabel("Standardized Effect Size (d)")
    ax.set_title(
        f"Forest Plot: MLLM Performance Meta-Analysis\n"
        f"I² = {overall_ma['I_sq']:.1f}%, τ² = {overall_ma['tau_sq']:.3f}, "
        f"Q({overall_ma['Q_df']}) = {overall_ma['Q']:.1f}",
        fontsize=11, fontweight="bold"
    )
    
    # Legend
    legend_elements = [
        mpatches.Patch(color="#e74c3c", label="RLHF"),
        mpatches.Patch(color="#3498db", label="Instruction Tuning"),
        mpatches.Patch(color="#2ecc71", label="Pretraining + Alignment"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    
    ax.set_ylim(-2.5, n + 0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 2: Funnel Plot
# ============================================================================

def plot_funnel(
    es_df: pd.DataFrame, 
    overall_ma: Dict, 
    egger_results: Dict,
    save_path: str
):
    """
    Create a funnel plot for publication bias assessment.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pooled = overall_ma["pooled_effect"]
    
    # Standard error vs effect size
    ax.scatter(
        es_df["d"], es_df["se"], 
        s=60, c="#3498db", edgecolors="black", linewidth=0.5, 
        alpha=0.8, zorder=3
    )
    
    # Pooled estimate line
    ax.axvline(x=pooled, color="#2c3e50", linestyle="--", linewidth=1, label=f"Pooled d = {pooled:.3f}")
    
    # Pseudo-confidence interval funnel
    se_range = np.linspace(0.001, max(es_df["se"]) * 1.3, 100)
    ci_lower = pooled - 1.96 * se_range
    ci_upper = pooled + 1.96 * se_range
    
    ax.fill_betweenx(
        se_range, ci_lower, ci_upper, 
        color="#3498db", alpha=0.1, label="95% Pseudo CI"
    )
    ax.plot(ci_lower, se_range, color="#3498db", linewidth=0.5, alpha=0.5)
    ax.plot(ci_upper, se_range, color="#3498db", linewidth=0.5, alpha=0.5)
    
    # Model labels
    for _, row in es_df.iterrows():
        ax.annotate(
            row["model"], (row["d"], row["se"]),
            fontsize=6, ha="center", va="bottom",
            xytext=(0, 4), textcoords="offset points", alpha=0.7
        )
    
    # Invert y-axis (convention: more precise at top)
    ax.invert_yaxis()
    ax.set_xlabel("Effect Size (d)")
    ax.set_ylabel("Standard Error")
    
    egger_p = egger_results["intercept_p"]
    egger_sig = "Yes" if egger_p < 0.05 else "No"
    ax.set_title(
        f"Funnel Plot for Publication Bias Assessment\n"
        f"Egger's test: intercept = {egger_results['intercept']:.3f}, "
        f"p = {egger_p:.3f} (Bias: {egger_sig})",
        fontsize=11, fontweight="bold"
    )
    
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 3: Benchmark Correlation Heatmap
# ============================================================================

def plot_correlation_heatmap(
    corr_mat: pd.DataFrame, 
    pval_mat: pd.DataFrame, 
    save_path: str
):
    """
    Create a heatmap of inter-benchmark Spearman correlations.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    
    # Create annotation matrix with significance markers
    annot = corr_mat.copy()
    annot_text = np.empty_like(corr_mat, dtype=object)
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat)):
            val = corr_mat.iloc[i, j]
            p = pval_mat.iloc[i, j]
            if np.isnan(val):
                annot_text[i, j] = ""
            elif p < 0.001:
                annot_text[i, j] = f"{val:.2f}***"
            elif p < 0.01:
                annot_text[i, j] = f"{val:.2f}**"
            elif p < 0.05:
                annot_text[i, j] = f"{val:.2f}*"
            else:
                annot_text[i, j] = f"{val:.2f}"
    
    sns.heatmap(
        corr_mat, mask=mask, annot=annot_text, fmt="",
        cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(
        "Inter-Benchmark Correlation Matrix (Spearman ρ)\n"
        "* p<0.05, ** p<0.01, *** p<0.001",
        fontsize=11, fontweight="bold"
    )
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 4: Scale-Performance Scatter Plot
# ============================================================================

def plot_scale_performance(
    df: pd.DataFrame, 
    scale_results: Dict, 
    save_path: str
):
    """
    Create a scatter plot of model scale vs. performance with regression lines.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color palette for benchmarks
    benchmark_colors = {
        "MMBench": "#e74c3c",
        "SEED-Bench": "#3498db",
        "MM-Vet": "#2ecc71",
        "MME": "#f39c12",
        "TextVQA": "#9b59b6",
        "POPE": "#1abc9c",
        "VQAv2": "#e67e22",
    }
    
    for bench, group in df.groupby("benchmark"):
        color = benchmark_colors.get(bench, "#95a5a6")
        ax.scatter(
            group["log_params"], group["normalized_score"],
            s=50, c=color, alpha=0.7, edgecolors="white", linewidth=0.5,
            label=bench, zorder=3
        )
    
    # Overall regression line
    overall = scale_results.get("overall", {})
    if overall:
        x_range = np.linspace(df["log_params"].min() - 0.1, df["log_params"].max() + 0.1, 100)
        y_pred = overall["intercept"] + overall["slope"] * x_range
        ax.plot(
            x_range, y_pred, 
            color="#2c3e50", linewidth=2, linestyle="--",
            label=f"Overall: r={overall['r']:.3f}, p={overall['p']:.3f}", 
            zorder=2
        )
    
    # Custom x-axis labels (show actual parameter counts)
    tick_values = [np.log10(x) for x in [1, 5, 10, 20, 50, 100, 500, 1000]]
    tick_labels = ["1B", "5B", "10B", "20B", "50B", "100B", "500B", "1T"]
    valid_ticks = [(v, l) for v, l in zip(tick_values, tick_labels) 
                   if df["log_params"].min() - 0.3 <= v <= df["log_params"].max() + 0.3]
    if valid_ticks:
        ax.set_xticks([v for v, _ in valid_ticks])
        ax.set_xticklabels([l for _, l in valid_ticks])
    
    ax.set_xlabel("Model Parameters (log scale)")
    ax.set_ylabel("Normalized Score [0, 1]")
    ax.set_title(
        "Model Scale vs. Benchmark Performance\n"
        f"Overall Pearson r = {overall.get('r', 0):.3f} "
        f"(p = {overall.get('p', 1):.4f}), n = {overall.get('n', 0)}",
        fontsize=11, fontweight="bold"
    )
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 5: Subgroup Forest Plot
# ============================================================================

def plot_subgroup_forest(
    subgroup_df: pd.DataFrame, 
    title: str, 
    save_path: str
):
    """
    Forest plot for subgroup meta-analysis results.
    """
    if subgroup_df.empty:
        print(f"  Skipping {save_path}: no subgroup data")
        return
    
    n = len(subgroup_df)
    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.8 + 1)))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    
    for i, (_, row) in enumerate(subgroup_df.iterrows()):
        ci_lower = row["pooled_ci_lower"]
        ci_upper = row["pooled_ci_upper"]
        
        ax.plot(
            [ci_lower, ci_upper], [i, i],
            color=colors[i], linewidth=2.5, zorder=2
        )
        ax.scatter(
            row["pooled_effect"], i,
            s=120, color=colors[i],
            edgecolors="black", linewidth=0.8, zorder=3,
            marker="D"
        )
        
        # Annotation
        ax.annotate(
            f"d = {row['pooled_effect']:.3f}\n"
            f"[{ci_lower:.3f}, {ci_upper:.3f}]\n"
            f"I² = {row['I_sq']:.1f}%, k = {row['n_models']}",
            xy=(ci_upper, i),
            fontsize=7, va="center",
            xytext=(8, 0), textcoords="offset points",
        )
    
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    
    labels = [str(row["subgroup"]) for _, row in subgroup_df.iterrows()]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Pooled Standardized Effect Size (d)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    ax.set_ylim(-0.5, n - 0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 6: Radar Chart of Top Models
# ============================================================================

def plot_radar_chart(
    df: pd.DataFrame, 
    top_n: int = 6,
    save_path: str = ""
):
    """
    Create a radar chart comparing top models across benchmarks.
    """
    # Select top models by average normalized score
    model_means = df.groupby("model")["normalized_score"].mean().sort_values(ascending=False)
    top_models = model_means.head(top_n).index.tolist()
    
    # Get benchmarks with enough coverage
    pivot = df[df["model"].isin(top_models)].pivot_table(
        index="model", columns="benchmark", values="normalized_score"
    )
    
    # Use only benchmarks where all top models have scores
    valid_benchmarks = pivot.columns[pivot.notna().sum() >= len(top_models) - 1].tolist()
    if len(valid_benchmarks) < 3:
        # Fallback: use benchmarks with most coverage
        valid_benchmarks = pivot.columns[pivot.notna().sum() >= 3].tolist()
    
    pivot = pivot[valid_benchmarks].fillna(0)
    
    n_benchmarks = len(valid_benchmarks)
    angles = np.linspace(0, 2 * np.pi, n_benchmarks, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))
    
    for idx, model in enumerate(top_models):
        if model in pivot.index:
            values = pivot.loc[model].values.tolist()
            values += values[:1]  # close
            
            ax.plot(angles, values, "o-", linewidth=2, label=model,
                   color=colors[idx], markersize=5)
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_benchmarks, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Top-{top_n} MLLM Performance Profiles\nAcross Benchmarks (Normalized Scores)",
        fontsize=11, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Figure 7: Box Plot by Training Strategy
# ============================================================================

def plot_training_strategy_box(
    df: pd.DataFrame, 
    kw_results: Dict,
    save_path: str
):
    """
    Box plot comparing performance distributions by training strategy.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    strategy_labels = {
        "RLHF": "RLHF",
        "instruction_tuning": "Instruction\nTuning",
        "pretraining_alignment": "Pretraining +\nAlignment",
    }
    
    df_plot = df.copy()
    df_plot["strategy_label"] = df_plot["training_strategy"].map(strategy_labels)
    
    palette = {
        "RLHF": "#e74c3c",
        "Instruction\nTuning": "#3498db",
        "Pretraining +\nAlignment": "#2ecc71",
    }
    
    order = ["RLHF", "Instruction\nTuning", "Pretraining +\nAlignment"]
    existing_order = [o for o in order if o in df_plot["strategy_label"].values]
    
    sns.boxplot(
        data=df_plot, x="strategy_label", y="normalized_score",
        order=existing_order, palette=palette,
        width=0.5, linewidth=1.5, ax=ax
    )
    
    sns.stripplot(
        data=df_plot, x="strategy_label", y="normalized_score",
        order=existing_order, color="black", alpha=0.4, 
        size=3, jitter=0.2, ax=ax
    )
    
    kw_H = kw_results.get("H", np.nan)
    kw_p = kw_results.get("p", np.nan)
    kw_eta = kw_results.get("eta_sq", np.nan)
    
    ax.set_xlabel("Training Strategy")
    ax.set_ylabel("Normalized Score [0, 1]")
    ax.set_title(
        f"Performance Distribution by Training Strategy\n"
        f"Kruskal-Wallis H = {kw_H:.2f}, p = {kw_p:.4f}, η² = {kw_eta:.3f}",
        fontsize=11, fontweight="bold"
    )
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Main: Generate All Figures
# ============================================================================

def generate_all_figures(results: Dict, output_dir: str):
    """
    Generate all publication figures.
    
    Parameters
    ----------
    results : dict
        Results from statistical_analysis.run_full_analysis().
    output_dir : str
        Directory to save figures.
    """
    ensure_dir(output_dir)
    
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    
    # Figure 1: Forest plot
    print("\n[1/7] Forest plot...")
    plot_forest(
        results["effect_sizes"],
        results["overall_meta_analysis"],
        os.path.join(output_dir, "fig1_forest_plot.png")
    )
    
    # Figure 2: Funnel plot
    print("[2/7] Funnel plot...")
    plot_funnel(
        results["effect_sizes"],
        results["overall_meta_analysis"],
        results["eggers_test"],
        os.path.join(output_dir, "fig2_funnel_plot.png")
    )
    
    # Figure 3: Correlation heatmap
    print("[3/7] Correlation heatmap...")
    plot_correlation_heatmap(
        results["benchmark_correlations"],
        results["benchmark_corr_pvalues"],
        os.path.join(output_dir, "fig3_correlation_heatmap.png")
    )
    
    # Figure 4: Scale-performance
    print("[4/7] Scale-performance scatter...")
    plot_scale_performance(
        results["data"],
        results["scale_regression"],
        os.path.join(output_dir, "fig4_scale_performance.png")
    )
    
    # Figure 5: Subgroup forest (training strategy)
    print("[5/7] Subgroup forest plot (training strategy)...")
    plot_subgroup_forest(
        results["subgroup_training"],
        "Subgroup Analysis by Training Strategy",
        os.path.join(output_dir, "fig5_subgroup_training.png")
    )
    
    # Figure 6: Radar chart
    print("[6/7] Radar chart...")
    plot_radar_chart(
        results["data"],
        top_n=6,
        save_path=os.path.join(output_dir, "fig6_radar_chart.png")
    )
    
    # Figure 7: Box plot
    print("[7/7] Training strategy box plot...")
    plot_training_strategy_box(
        results["data"],
        results["kw_training"],
        os.path.join(output_dir, "fig7_training_boxplot.png")
    )
    
    print("\n✓ All 7 figures generated successfully.")
