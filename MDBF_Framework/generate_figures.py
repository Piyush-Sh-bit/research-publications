"""
=============================================================
Publication-Quality Figure Generation Script
A Multi-Dimensional Benchmarking Framework for MLLMs
=============================================================
Imports computed results from compute_results.py pipeline.
All derived metrics are COMPUTED, not hardcoded.

Generates: radar charts, bar charts, heatmaps, gap & error analysis
Output: figures/ directory with PDF + PNG files for LaTeX inclusion

Requirements: pip install matplotlib seaborn numpy scipy
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings

# Import our computation pipeline
from compute_results import (
    MODELS, RAW_SCORES, DIMENSION_CONFIG, ERROR_TAXONOMY,
    ERROR_CATEGORIES, MODEL_TYPE,
    normalize_scores, compute_dimension_scores,
    compute_overall_scores, compute_pgr, run_pipeline
)

warnings.filterwarnings("ignore")

# ===================== CONFIGURATION =====================
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Create output directory
os.makedirs("figures", exist_ok=True)

# ===================== COMPUTE ALL DATA =====================
# Run the full pipeline — everything is computed from raw scores
print("Running MDBF computation pipeline...")
pipeline_results = run_pipeline()

normalized_scores = pipeline_results["normalized"]
dimension_scores = pipeline_results["dimension_scores"]
overall_scores = pipeline_results["overall_scores"]
pgr_data = pipeline_results["pgr"]

# Display labels
models_short = ["GPT-4V", "Gemini Pro", "LLaVA-1.5", "InstructBLIP", "Qwen-VL", "CogVLM"]
models_wrap = ["GPT-4V", "Gemini Pro\nVision", "LLaVA-1.5\n13B", "InstructBLIP", "Qwen-VL\nChat", "CogVLM"]

dimension_labels = [
    "Visual Perception",
    "Reasoning & Inference",
    "OCR & Text Understanding",
    "Knowledge Integration",
    "Robustness & Hallucination\nResistance"
]

# Extract computed dimension data as list-of-lists for plotting
dim_keys = list(dimension_scores.keys())
dimension_data = [dimension_scores[k] for k in dim_keys]

# Color palette - professional and publication-ready
colors = {
    "GPT-4V":       "#2E86AB",
    "Gemini Pro":   "#A23B72",
    "LLaVA-1.5":    "#F18F01",
    "InstructBLIP":  "#C73E1D",
    "Qwen-VL":      "#3B1F2B",
    "CogVLM":       "#44BBA4",
}
color_list = list(colors.values())


# ===================== FIGURE 1: RADAR CHART =====================
def plot_radar_chart():
    """Generate radar chart comparing all models across 5 dimensions."""
    N = len(dimension_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimension_labels, fontsize=9, fontweight="bold")

    ax.set_rlabel_position(30)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="grey")
    ax.set_ylim(0, 110)

    for idx, model in enumerate(models_short):
        values = [dimension_data[d][idx] for d in range(N)]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model,
                color=color_list[idx], markersize=5)
        ax.fill(angles, values, alpha=0.08, color=color_list[idx])

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
              frameon=True, fancybox=True, shadow=True)
    ax.set_title("Multi-Dimensional Competency Profiles\nof Evaluated MLLMs",
                 pad=25, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/radar_chart.pdf", format="pdf")
    plt.savefig("figures/radar_chart.png", format="png")
    plt.close()
    print("✓ Generated: figures/radar_chart.pdf")


# ===================== FIGURE 2: BENCHMARK BAR CHART =====================
def plot_benchmark_bars():
    """Generate grouped bar chart for raw benchmark scores."""
    benchmarks = list(RAW_SCORES.keys())
    n_models = len(models_short)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, bench in enumerate(benchmarks):
        ax = axes[i]
        scores = RAW_SCORES[bench]
        bars = ax.bar(range(n_models), scores, color=color_list, width=0.7,
                      edgecolor="white", linewidth=0.5)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{score:.1f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")

        ax.set_title(bench, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models_short, rotation=45, ha="right", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        min_score = min(scores) * 0.85
        max_score = max(scores) * 1.08
        ax.set_ylim(min_score, max_score)

    plt.suptitle("Per-Benchmark Performance Comparison Across Six MLLMs",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/benchmark_bars.pdf", format="pdf")
    plt.savefig("figures/benchmark_bars.png", format="png")
    plt.close()
    print("✓ Generated: figures/benchmark_bars.pdf")


# ===================== FIGURE 3: DIMENSION HEATMAP =====================
def plot_dimension_heatmap():
    """Generate heatmap of COMPUTED normalized dimension scores."""
    dim_labels_clean = [
        "Visual Perception",
        "Reasoning & Inference",
        "OCR & Text Understanding",
        "Knowledge Integration",
        "Robustness & Hallucination"
    ]

    # dimension_data is computed from the pipeline, not hardcoded
    data = np.array(dimension_data).T  # models x dimensions
    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(dim_labels_clean)))
    ax.set_xticklabels(dim_labels_clean, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(models_short)))
    ax.set_yticklabels(models_short, fontsize=10)

    for i in range(len(models_short)):
        for j in range(len(dim_labels_clean)):
            val = data[i, j]
            text_color = "white" if val > 65 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Normalized Score (0–100)", fontsize=10)

    ax.set_title("Normalized Dimension Scores — Heatmap Visualization",
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig("figures/dimension_heatmap.pdf", format="pdf")
    plt.savefig("figures/dimension_heatmap.png", format="png")
    plt.close()
    print("✓ Generated: figures/dimension_heatmap.pdf")


# ===================== FIGURE 4: PGR BAR CHART =====================
def plot_pgr_analysis():
    """Generate COMPUTED Performance Gap Ratio analysis chart."""
    dims = list(pgr_data.keys())
    dims_short = ["D1: Visual\nPerception", "D2: Reasoning\n& Inference",
                  "D3: OCR & Text\nUnderstanding", "D4: Knowledge\nIntegration",
                  "D5: Robustness"]
    pgr_values = [pgr_data[d]["pgr"] for d in dims]

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#44BBA4"]
    bars = ax.barh(dims_short, pgr_values, color=bar_colors, height=0.6,
                   edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, pgr_values):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Performance Gap Ratio (%)", fontsize=12, fontweight="bold")
    ax.set_title("Proprietary vs. Open-Source Performance Gap by Dimension",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, max(pgr_values) * 1.3)
    ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/pgr_analysis.pdf", format="pdf")
    plt.savefig("figures/pgr_analysis.png", format="png")
    plt.close()
    print("✓ Generated: figures/pgr_analysis.pdf")


# ===================== FIGURE 5: ERROR TAXONOMY =====================
def plot_error_taxonomy():
    """Generate grouped bar chart of error taxonomy distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))

    error_types_wrap = ["Visual\nMisrecognition", "Reasoning\nFailure", "Knowledge\nGap",
                        "Hallucination", "OCR\nFailure", "Format/Parse\nError"]

    x = np.arange(len(error_types_wrap))
    width = 0.13
    multiplier = 0

    error_short_models = {
        "GPT-4V": ERROR_TAXONOMY["GPT-4V"],
        "Gemini Pro": ERROR_TAXONOMY["Gemini Pro Vision"],
        "LLaVA-1.5": ERROR_TAXONOMY["LLaVA-1.5-13B"],
        "InstructBLIP": ERROR_TAXONOMY["InstructBLIP"],
        "Qwen-VL": ERROR_TAXONOMY["Qwen-VL-Chat"],
        "CogVLM": ERROR_TAXONOMY["CogVLM"],
    }

    for model_name, errors in error_short_models.items():
        offset = width * multiplier
        bars = ax.bar(x + offset, errors, width, label=model_name,
                      color=colors[model_name], edgecolor="white", linewidth=0.5)
        multiplier += 1

    ax.set_xlabel("Error Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion (%)", fontsize=12, fontweight="bold")
    ax.set_title("Error Distribution Taxonomy Across Evaluated MLLMs",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(error_types_wrap, fontsize=9)
    ax.legend(loc="upper right", frameon=True, fancybox=True, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 38)

    plt.tight_layout()
    plt.savefig("figures/error_taxonomy.pdf", format="pdf")
    plt.savefig("figures/error_taxonomy.png", format="png")
    plt.close()
    print("✓ Generated: figures/error_taxonomy.pdf")


# ===================== FIGURE 6: OVERALL RANKING =====================
def plot_overall_ranking():
    """Generate COMPUTED overall composite score ranking chart."""
    # overall_scores is computed from pipeline, not hardcoded
    sorted_idx = np.argsort(overall_scores)[::-1]

    sorted_models = [models_short[i] for i in sorted_idx]
    sorted_scores = [overall_scores[i] for i in sorted_idx]
    sorted_colors = [color_list[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(range(len(sorted_models)), sorted_scores,
                   color=sorted_colors, height=0.6, edgecolor="white", linewidth=1.5)

    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", ha="left", va="center", fontsize=12, fontweight="bold")

    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(sorted_models, fontsize=11)
    ax.set_xlabel("Normalized Composite Score (0–100)", fontsize=12, fontweight="bold")
    ax.set_title("Overall MDBF Composite Score Ranking",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/overall_ranking.pdf", format="pdf")
    plt.savefig("figures/overall_ranking.png", format="png")
    plt.close()
    print("✓ Generated: figures/overall_ranking.pdf")


# ===================== FIGURE 7: CROSS-DIMENSION CORRELATION =====================
def plot_correlation_heatmap():
    """Generate Spearman correlation heatmap between dimensions."""
    dim_names_short = ["D1: Percep.", "D2: Reason.", "D3: OCR",
                       "D4: Knowl.", "D5: Robust."]
    dim_keys_list = list(dimension_scores.keys())

    from scipy import stats as sp_stats
    n = len(dim_keys_list)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, _ = sp_stats.spearmanr(
                dimension_scores[dim_keys_list[i]],
                dimension_scores[dim_keys_list[j]]
            )
            corr_matrix[i][j] = round(rho, 3)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(dim_names_short, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(dim_names_short, fontsize=9)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr_matrix[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=10)

    ax.set_title("Cross-Dimension Spearman Rank Correlations",
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.pdf", format="pdf")
    plt.savefig("figures/correlation_heatmap.png", format="png")
    plt.close()
    print("✓ Generated: figures/correlation_heatmap.pdf")


# ===================== MAIN =====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generating publication-quality figures for MDBF paper")
    print("All values are COMPUTED from raw benchmark scores")
    print("=" * 60)

    plot_radar_chart()
    plot_benchmark_bars()
    plot_dimension_heatmap()
    plot_pgr_analysis()
    plot_error_taxonomy()
    plot_overall_ranking()
    plot_correlation_heatmap()

    print("=" * 60)
    print("All figures generated successfully!")
    print("Output directory: figures/")
    print("Files: radar_chart.pdf, benchmark_bars.pdf,")
    print("       dimension_heatmap.pdf, pgr_analysis.pdf,")
    print("       error_taxonomy.pdf, overall_ranking.pdf,")
    print("       correlation_heatmap.pdf")
    print("=" * 60)
