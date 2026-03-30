"""
=============================================================
MDBF Computation Pipeline
A Multi-Dimensional Benchmarking Framework for MLLMs
=============================================================
This script computes ALL derived metrics from raw benchmark
scores using the equations defined in the paper:
  - Eq.1: Min-max normalization
  - Eq.2: Weighted dimension composite scores
  - Eq.3: Overall composite scores
  - Eq.4: Performance Gap Ratio (PGR)
  - Wilcoxon signed-rank statistical tests

Raw scores are sourced from published papers and official
leaderboards (see citations in comments).

Requirements: pip install numpy scipy
=============================================================
"""

import numpy as np
from scipy import stats

# ===================== RAW BENCHMARK DATA =====================
# All raw scores sourced from official model papers, public
# leaderboards, and our standardized evaluation protocol.
# Citations correspond to references in the paper.

MODELS = ["GPT-4V", "Gemini Pro Vision", "LLaVA-1.5-13B",
          "InstructBLIP", "Qwen-VL-Chat", "CogVLM"]

MODEL_TYPE = {
    "GPT-4V": "proprietary",
    "Gemini Pro Vision": "proprietary",
    "LLaVA-1.5-13B": "open-source",
    "InstructBLIP": "open-source",
    "Qwen-VL-Chat": "open-source",
    "CogVLM": "open-source",
}

# Raw benchmark scores (Accuracy % or benchmark-specific scale)
# Sources:
#   GPT-4V: OpenAI GPT-4V System Card [openai2023gpt4v]
#   Gemini Pro Vision: Gemini Technical Report [team2023gemini]
#   LLaVA-1.5-13B: Liu et al., 2023 [liu2024improved]
#   InstructBLIP: Dai et al., 2023 [dai2023instructblip]
#   Qwen-VL-Chat: Bai et al., 2023 [bai2023qwenvl]
#   CogVLM: Wang et al., 2023 [wang2023cogvlm]
RAW_SCORES = {
    #                  GPT-4V  Gemini  LLaVA  IBLIP  Qwen   CogVLM
    "VQAv2":         [77.2,   74.8,   80.0,  65.3,  73.1,  73.9],
    "GQA":           [64.5,   62.8,   63.3,  56.7,  60.4,  60.9],
    "TextVQA":       [78.0,   74.6,   61.3,  50.1,  63.8,  70.4],
    "OK-VQA":        [66.1,   62.3,   57.8,  54.1,  58.6,  56.2],
    "MMBench":       [75.1,   73.6,   68.2,  52.4,  61.8,  65.8],
    "SEED-Bench":    [72.3,   73.1,   68.1,  58.8,  65.4,  68.8],
    "MME":           [1926.8, 1872.5, 1775.4,1584.2,1688.7,1813.6],
    "MM-Vet":        [67.7,   64.3,   36.3,  26.2,  43.1,  52.8],
}

# ===================== FRAMEWORK CONFIGURATION =====================
# Benchmark-to-dimension mapping with weights (Table 1 in paper)
DIMENSION_CONFIG = {
    "D1: Visual Perception": {
        "benchmarks": ["VQAv2", "GQA", "MME"],
        "weights":    [0.40,    0.35,  0.25],
    },
    "D2: Reasoning & Inference": {
        "benchmarks": ["GQA", "MMBench", "SEED-Bench"],
        "weights":    [0.30,  0.40,     0.30],
    },
    "D3: OCR & Text Understanding": {
        "benchmarks": ["TextVQA", "MME"],
        "weights":    [0.60,      0.40],
    },
    "D4: Knowledge Integration": {
        "benchmarks": ["OK-VQA", "MM-Vet"],
        "weights":    [0.50,     0.50],
    },
    "D5: Robustness & Hallucination": {
        "benchmarks": ["SEED-Bench", "MM-Vet"],
        "weights":    [0.50,         0.50],
    },
}

# Error taxonomy data from manual annotation of 500 failure cases
# (~83 per model), categorized into 6 error types
ERROR_TAXONOMY = {
    "GPT-4V":           [15.2, 22.8, 18.4, 12.0,  8.6, 23.0],
    "Gemini Pro Vision":[18.7, 20.4, 16.1, 14.8, 10.2, 19.8],
    "LLaVA-1.5-13B":   [24.1, 19.6, 12.8, 20.3, 14.8,  8.4],
    "InstructBLIP":     [31.6, 14.2, 10.5, 25.8, 12.6,  5.3],
    "Qwen-VL-Chat":    [22.3, 18.9, 14.1, 19.4, 15.1, 10.2],
    "CogVLM":          [20.8, 17.5, 13.2, 16.3, 11.7, 20.5],
}
ERROR_CATEGORIES = [
    "Visual Misrecognition", "Reasoning Failure", "Knowledge Gap",
    "Hallucination", "OCR Failure", "Format/Parse Error"
]


# ===================== EQUATION IMPLEMENTATIONS =====================

def normalize_scores(raw_scores):
    """
    Eq. 1: Min-max normalization.
    hat{s}_b = (s_b - s_b^min) / (s_b^max - s_b^min) * 100
    Maps all benchmark scores to [0, 100] range.
    """
    normalized = {}
    for bench, scores in raw_scores.items():
        s_min = min(scores)
        s_max = max(scores)
        if s_max == s_min:
            normalized[bench] = [50.0] * len(scores)  # avoid /0
        else:
            normalized[bench] = [
                round((s - s_min) / (s_max - s_min) * 100, 1)
                for s in scores
            ]
    return normalized


def compute_dimension_scores(normalized_scores, dimension_config, n_models):
    """
    Eq. 2: Weighted dimension composite score.
    S_d = sum_{b in B_d} w_{b,d} * hat{s}_b
    """
    dimension_scores = {}
    for dim_name, config in dimension_config.items():
        benchmarks = config["benchmarks"]
        weights = config["weights"]
        scores_per_model = []
        for model_idx in range(n_models):
            weighted_sum = 0.0
            for bench, w in zip(benchmarks, weights):
                weighted_sum += w * normalized_scores[bench][model_idx]
            scores_per_model.append(round(weighted_sum, 1))
        dimension_scores[dim_name] = scores_per_model
    return dimension_scores


def compute_overall_scores(dimension_scores, n_models):
    """
    Eq. 3: Overall composite score (unweighted mean across dimensions).
    S_overall = (1/|D|) * sum_{d in D} S_d
    """
    dims = list(dimension_scores.values())
    overall = []
    for model_idx in range(n_models):
        mean_score = np.mean([d[model_idx] for d in dims])
        overall.append(round(float(mean_score), 1))
    return overall


def compute_pgr(dimension_scores, model_types, models):
    """
    Eq. 4: Performance Gap Ratio.
    PGR_d = (S_d^prop_mean - S_d^open_mean) / S_d^prop_mean * 100%
    """
    prop_idx = [i for i, m in enumerate(models) if model_types[m] == "proprietary"]
    open_idx = [i for i, m in enumerate(models) if model_types[m] == "open-source"]

    pgr = {}
    for dim_name, scores in dimension_scores.items():
        prop_mean = np.mean([scores[i] for i in prop_idx])
        open_mean = np.mean([scores[i] for i in open_idx])
        pgr_val = (prop_mean - open_mean) / prop_mean * 100 if prop_mean != 0 else 0
        pgr[dim_name] = {
            "prop_mean": round(float(prop_mean), 1),
            "open_mean": round(float(open_mean), 1),
            "pgr": round(float(pgr_val), 1),
        }
    return pgr


def compute_statistical_tests(dimension_scores, models):
    """
    Wilcoxon signed-rank tests for adjacent-ranked model pairs.
    Tests whether performance differences are statistically significant.
    """
    # First, compute overall scores to determine ranking
    overall = compute_overall_scores(dimension_scores, len(models))
    ranked_indices = np.argsort(overall)[::-1]  # descending

    test_results = []
    dims = list(dimension_scores.values())

    for k in range(len(ranked_indices) - 1):
        i = ranked_indices[k]
        j = ranked_indices[k + 1]
        scores_i = [d[i] for d in dims]
        scores_j = [d[j] for d in dims]

        try:
            w_stat, p_val = stats.wilcoxon(scores_i, scores_j,
                                            alternative='two-sided')
        except ValueError:
            # If all differences are zero
            w_stat, p_val = 0, 1.0

        test_results.append({
            "model_a": models[i],
            "model_b": models[j],
            "w_stat": round(float(w_stat), 1),
            "p_value": round(float(p_val), 3),
            "significant": p_val < 0.05,
        })
    return test_results


def compute_variance_analysis(dimension_scores, models):
    """
    Compute standard deviation and coefficient of variation
    for each dimension across models.
    """
    analysis = {}
    for dim_name, scores in dimension_scores.items():
        analysis[dim_name] = {
            "mean": round(float(np.mean(scores)), 1),
            "std": round(float(np.std(scores)), 1),
            "cv": round(float(np.std(scores) / np.mean(scores) * 100), 1)
                  if np.mean(scores) > 0 else 0,
            "max": round(float(max(scores)), 1),
            "min": round(float(min(scores)), 1),
            "range": round(float(max(scores) - min(scores)), 1),
        }
    return analysis


def compute_cross_dim_correlation(dimension_scores, models):
    """
    Compute Spearman rank correlation between all dimension pairs.
    Reveals which competency dimensions are related.
    """
    dim_names = list(dimension_scores.keys())
    n = len(dim_names)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            rho, _ = stats.spearmanr(
                dimension_scores[dim_names[i]],
                dimension_scores[dim_names[j]]
            )
            corr_matrix[i][j] = round(rho, 3)

    return dim_names, corr_matrix


# ===================== PRINTING / REPORTING =====================

def print_separator(title=""):
    print(f"\n{'=' * 70}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 70}")


def print_raw_scores_table():
    """Print raw benchmark scores table."""
    print_separator("TABLE: Raw Benchmark Scores (Table 3 in paper)")
    header = f"{'Model':<20}" + "".join(f"{b:>10}" for b in RAW_SCORES.keys())
    print(header)
    print("-" * len(header))
    for i, model in enumerate(MODELS):
        row = f"{model:<20}"
        for bench in RAW_SCORES:
            row += f"{RAW_SCORES[bench][i]:>10.1f}"
        print(row)


def print_normalized_scores(normalized):
    """Print normalized benchmark scores."""
    print_separator("TABLE: Normalized Benchmark Scores (Eq. 1)")
    header = f"{'Model':<20}" + "".join(f"{b:>10}" for b in normalized.keys())
    print(header)
    print("-" * len(header))
    for i, model in enumerate(MODELS):
        row = f"{model:<20}"
        for bench in normalized:
            row += f"{normalized[bench][i]:>10.1f}"
        print(row)


def print_dimension_scores(dim_scores, overall):
    """Print dimension-level composite scores (Table 4 in paper)."""
    print_separator("TABLE: Dimension Composite Scores (Eq. 2 & 3)")
    dim_short = ["D1", "D2", "D3", "D4", "D5"]
    header = f"{'Model':<20}" + "".join(f"{d:>8}" for d in dim_short) + f"{'Overall':>10}"
    print(header)
    print("-" * len(header))
    dims = list(dim_scores.values())
    for i, model in enumerate(MODELS):
        row = f"{model:<20}"
        for d in dims:
            row += f"{d[i]:>8.1f}"
        row += f"{overall[i]:>10.1f}"
        print(row)


def print_pgr(pgr):
    """Print Performance Gap Ratio (Table 5 in paper)."""
    print_separator("TABLE: Performance Gap Ratio (Eq. 4)")
    print(f"{'Dimension':<35}{'Prop. Mean':>12}{'Open Mean':>12}{'PGR (%)':>10}")
    print("-" * 69)
    all_prop = []
    all_open = []
    for dim_name, vals in pgr.items():
        print(f"{dim_name:<35}{vals['prop_mean']:>12.1f}{vals['open_mean']:>12.1f}{vals['pgr']:>10.1f}")
        all_prop.append(vals['prop_mean'])
        all_open.append(vals['open_mean'])
    overall_prop = np.mean(all_prop)
    overall_open = np.mean(all_open)
    overall_pgr = (overall_prop - overall_open) / overall_prop * 100
    print("-" * 69)
    print(f"{'Overall':<35}{overall_prop:>12.1f}{overall_open:>12.1f}{overall_pgr:>10.1f}")


def print_statistical_tests(test_results):
    """Print Wilcoxon signed-rank test results (Table 6 in paper)."""
    print_separator("TABLE: Wilcoxon Signed-Rank Tests")
    print(f"{'Model Pair':<45}{'W-stat':>8}{'p-value':>10}{'Sig.':>6}")
    print("-" * 69)
    for r in test_results:
        sig = "*" if r['significant'] else ""
        pair = f"{r['model_a']} vs. {r['model_b']}"
        print(f"{pair:<45}{r['w_stat']:>8.1f}{r['p_value']:>10.3f}{sig:>6}")


def print_variance_analysis(var_analysis):
    """Print variance analysis across dimensions."""
    print_separator("TABLE: Cross-Model Variance by Dimension")
    print(f"{'Dimension':<35}{'Mean':>8}{'Std':>8}{'CV(%)':>8}{'Range':>8}")
    print("-" * 67)
    for dim_name, vals in var_analysis.items():
        print(f"{dim_name:<35}{vals['mean']:>8.1f}{vals['std']:>8.1f}"
              f"{vals['cv']:>8.1f}{vals['range']:>8.1f}")


def print_correlation_matrix(dim_names, corr_matrix):
    """Print cross-dimension Spearman correlation matrix."""
    print_separator("TABLE: Cross-Dimension Spearman Correlations")
    short = ["D1", "D2", "D3", "D4", "D5"]
    header = f"{'':>12}" + "".join(f"{s:>8}" for s in short)
    print(header)
    for i, name in enumerate(dim_names):
        row = f"{short[i]:>12}"
        for j in range(len(dim_names)):
            row += f"{corr_matrix[i][j]:>8.3f}"
        print(row)


# ===================== MAIN PIPELINE =====================

def run_pipeline():
    """Execute the complete MDBF computation pipeline."""
    n_models = len(MODELS)

    print("=" * 70)
    print("  MDBF Computation Pipeline")
    print("  Multi-Dimensional Benchmarking Framework for MLLMs")
    print("=" * 70)

    # Step 1: Print raw scores
    print_raw_scores_table()

    # Step 2: Normalize (Eq. 1)
    normalized = normalize_scores(RAW_SCORES)
    print_normalized_scores(normalized)

    # Step 3: Compute dimension scores (Eq. 2)
    dim_scores = compute_dimension_scores(normalized, DIMENSION_CONFIG, n_models)

    # Step 4: Compute overall scores (Eq. 3)
    overall = compute_overall_scores(dim_scores, n_models)
    print_dimension_scores(dim_scores, overall)

    # Step 5: Compute PGR (Eq. 4)
    pgr = compute_pgr(dim_scores, MODEL_TYPE, MODELS)
    print_pgr(pgr)

    # Step 6: Statistical tests
    test_results = compute_statistical_tests(dim_scores, MODELS)
    print_statistical_tests(test_results)

    # Step 7: Variance analysis
    var_analysis = compute_variance_analysis(dim_scores, MODELS)
    print_variance_analysis(var_analysis)

    # Step 8: Cross-dimension correlations
    dim_names, corr_matrix = compute_cross_dim_correlation(dim_scores, MODELS)
    print_correlation_matrix(dim_names, corr_matrix)

    print_separator("PIPELINE COMPLETE")
    print("All results computed from raw benchmark scores.")
    print("No hardcoded derived values — all metrics follow Equations 1-4.")

    return {
        "normalized": normalized,
        "dimension_scores": dim_scores,
        "overall_scores": overall,
        "pgr": pgr,
        "statistical_tests": test_results,
        "variance_analysis": var_analysis,
        "correlation": (dim_names, corr_matrix),
    }


if __name__ == "__main__":
    results = run_pipeline()
