# Author: Piyush Sharma
"""
robustness_analysis.py
======================
Robustness and sensitivity analyses to address meta-analysis shortcomings.

Implements:
  1. Hedges' g effect sizes (classical effect size metric)
  2. Leave-one-out sensitivity analysis
  3. Bootstrap confidence intervals for benchmark correlations
    3b. Leave-one-benchmark-out multilevel sensitivity
  4. Sensitivity analysis excluding proprietary models
  5. Galbraith (radial) plot data
  6. Influence diagnostics (DFBETAS analog)
  7. Efficiency/Pareto frontier analysis
  8. Minimum sample threshold for correlations
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# 1. Hedges' g Effect Sizes
# ============================================================================

def compute_hedges_g(zscores: np.ndarray, n_benchmarks: int) -> Tuple[float, float]:
    """
    Compute Hedges' g (bias-corrected standardized mean difference).

    Applies the small-sample correction factor J to convert from
    Cohen's d to Hedges' g, using df = n_benchmarks - 1.

    Parameters
    ----------
    zscores : array-like
        Z-scores across benchmarks for a single model.
    n_benchmarks : int
        Number of benchmarks the model was evaluated on.

    Returns
    -------
    Tuple[float, float]
        (hedges_g, variance_g)
    """
    d = np.mean(zscores)
    n = len(zscores)

    # Standard error of d
    if n > 1:
        se_d = np.std(zscores, ddof=1) / np.sqrt(n)
    else:
        se_d = 0.3  # conservative

    se_d = max(se_d, 0.05)

    # Hedges' correction factor J (exact)
    # J ≈ 1 - 3/(4*df - 1) where df = n - 1
    df = max(n - 1, 1)
    J = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 0.75

    g = J * d
    var_g = (J ** 2) * (se_d ** 2)

    return g, var_g


def compute_all_hedges_g(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Hedges' g for all models, alongside standard z-score effect sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized data with 'model' and 'zscore' columns.

    Returns
    -------
    pd.DataFrame
        Effect sizes with both d (z-score) and g (Hedges') columns.
    """
    results = []
    for model, group in df.groupby("model"):
        zscores = group["zscore"].values
        n = len(zscores)
        d = np.mean(zscores)

        if n > 1:
            se_d = np.std(zscores, ddof=1) / np.sqrt(n)
        else:
            se_d = 0.3
        se_d = max(se_d, 0.05)

        g, var_g = compute_hedges_g(zscores, n)
        se_g = np.sqrt(var_g)

        meta = group.iloc[0]
        results.append({
            "model": model,
            "d": d,
            "se_d": se_d,
            "var_d": se_d ** 2,
            "hedges_g": g,
            "se_g": se_g,
            "var_g": var_g,
            "ci_g_lower": g - 1.96 * se_g,
            "ci_g_upper": g + 1.96 * se_g,
            "n_benchmarks": n,
            "params_b": meta["params_b"],
            "training_strategy": meta["training_strategy"],
            "encoder_family": meta["encoder_family"],
        })

    return pd.DataFrame(results).sort_values("hedges_g", ascending=True).reset_index(drop=True)


# ============================================================================
# 2. Leave-One-Out Sensitivity Analysis
# ============================================================================

def leave_one_out_analysis(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    model_names: List[str]
) -> pd.DataFrame:
    """
    Perform leave-one-out sensitivity analysis.

    Removes each model in turn and re-runs the random-effects meta-analysis
    to assess the influence of each individual model on the pooled estimate.

    Parameters
    ----------
    effect_sizes : array-like
        Model effect sizes.
    variances : array-like
        Model variances.
    model_names : list
        Model name strings.

    Returns
    -------
    pd.DataFrame
        Leave-one-out results with pooled estimates excluding each model.
    """
    from statistical_analysis import random_effects_meta_analysis

    d = np.asarray(effect_sizes, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = len(d)

    # Full analysis
    full_ma = random_effects_meta_analysis(d, v)

    results = []
    for i in range(k):
        # Remove model i
        d_loo = np.delete(d, i)
        v_loo = np.delete(v, i)

        ma_loo = random_effects_meta_analysis(d_loo, v_loo)

        # Influence: change in pooled estimate
        delta = full_ma["pooled_effect"] - ma_loo["pooled_effect"]

        results.append({
            "excluded_model": model_names[i],
            "pooled_d": ma_loo["pooled_effect"],
            "pooled_se": ma_loo["pooled_se"],
            "ci_lower": ma_loo["pooled_ci_lower"],
            "ci_upper": ma_loo["pooled_ci_upper"],
            "I_sq": ma_loo["I_sq"],
            "tau_sq": ma_loo["tau_sq"],
            "Q": ma_loo["Q"],
            "delta_d": delta,
            "abs_delta": abs(delta),
        })

    return pd.DataFrame(results).sort_values("abs_delta", ascending=False)


# ============================================================================
# 3. Bootstrap CIs for Benchmark Correlations
# ============================================================================

def bootstrap_correlation_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for pairwise Spearman
    correlations between benchmarks.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized data with 'model', 'benchmark', 'normalized_score'.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95).
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Bootstrap CI results for each benchmark pair.
    """
    rng = np.random.RandomState(random_state)

    pivot = df.pivot_table(
        index="model", columns="benchmark", values="normalized_score"
    )
    benchmarks = pivot.columns.tolist()

    results = []
    alpha = 1.0 - ci_level

    for i in range(len(benchmarks)):
        for j in range(i + 1, len(benchmarks)):
            b1, b2 = benchmarks[i], benchmarks[j]
            valid = pivot[[b1, b2]].dropna()
            n_obs = len(valid)

            if n_obs < 4:
                results.append({
                    "benchmark_1": b1, "benchmark_2": b2,
                    "n": n_obs, "rho": np.nan,
                    "ci_lower": np.nan, "ci_upper": np.nan,
                    "reliable": False,
                })
                continue

            # Point estimate
            rho, p = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])

            # Bootstrap
            boot_rhos = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n_obs, size=n_obs, replace=True)
                sample = valid.iloc[idx]
                try:
                    r, _ = stats.spearmanr(sample.iloc[:, 0], sample.iloc[:, 1])
                    if not np.isnan(r):
                        boot_rhos.append(r)
                except:
                    pass

            if len(boot_rhos) > 100:
                boot_rhos = np.array(boot_rhos)
                ci_lo = np.percentile(boot_rhos, 100 * alpha / 2)
                ci_hi = np.percentile(boot_rhos, 100 * (1 - alpha / 2))
                ci_width = ci_hi - ci_lo
            else:
                ci_lo, ci_hi, ci_width = np.nan, np.nan, np.nan

            results.append({
                "benchmark_1": b1, "benchmark_2": b2,
                "n": n_obs, "rho": rho,
                "ci_lower": ci_lo, "ci_upper": ci_hi,
                "ci_width": ci_width,
                "reliable": n_obs >= 5 and ci_width < 0.8 if not np.isnan(ci_width) else False,
            })

    return pd.DataFrame(results)


# ============================================================================
# 3b. Leave-One-Benchmark-Out (Primary Multilevel) Sensitivity
# ============================================================================

def leave_one_benchmark_out_multilevel(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Perform leave-one-benchmark-out sensitivity analysis for the
    primary multilevel model.

    Re-fits the primary mixed-effects specification after excluding each
    benchmark and tracks how the scale coefficient changes.

    Parameters
    ----------
    df_norm : pd.DataFrame
        Normalized observation-level data.

    Returns
    -------
    pd.DataFrame
        Per-benchmark exclusion results with coefficient shifts.
    """
    from multilevel_analysis import fit_multilevel_model

    formula = "zscore ~ log_params + C(training_strategy, Treatment(reference='instruction_tuning'))"
    full = fit_multilevel_model(df_norm, formula=formula, group_var="model")
    full_fe = full["fixed_effects"].set_index("parameter")
    full_scale = float(full_fe.loc["log_params", "coefficient"])

    rows = []
    for bench in sorted(df_norm["benchmark"].unique()):
        sub = df_norm[df_norm["benchmark"] != bench].copy()
        fit = fit_multilevel_model(sub, formula=formula, group_var="model")
        fe = fit["fixed_effects"].set_index("parameter")

        scale_coef = float(fe.loc["log_params", "coefficient"])
        scale_se = float(fe.loc["log_params", "se"])
        scale_p = float(fe.loc["log_params", "p"])
        delta = scale_coef - full_scale

        rows.append({
            "excluded_benchmark": bench,
            "n_obs": int(fit["n_obs"]),
            "n_groups": int(fit["n_groups"]),
            "scale_coef": scale_coef,
            "scale_se": scale_se,
            "scale_p": scale_p,
            "delta_scale_coef": delta,
            "abs_delta_scale_coef": abs(delta),
            "sigma2_between": float(fit["sigma2_between"]),
            "sigma2_within": float(fit["sigma2_within"]),
            "icc": float(fit["icc"]),
            "aic": float(fit["aic"]),
            "bic": float(fit["bic"]),
        })

    out = pd.DataFrame(rows).sort_values("abs_delta_scale_coef", ascending=False)
    out.attrs["full_scale_coef"] = full_scale
    out.attrs["full_icc"] = float(full["icc"])
    return out


# ============================================================================
# 4. Sensitivity Analysis Excluding Proprietary Models
# ============================================================================

def sensitivity_excluding_proprietary(es_df: pd.DataFrame) -> Dict:
    """
    Re-run meta-analysis excluding proprietary models (GPT-4V, Gemini)
    to assess their influence on results.

    Parameters
    ----------
    es_df : pd.DataFrame
        Effect size DataFrame.

    Returns
    -------
    dict
        Meta-analysis results for open-source models only, plus comparison.
    """
    from statistical_analysis import (
        random_effects_meta_analysis,
        meta_regression,
        subgroup_meta_analysis,
    )

    # Identify proprietary models
    proprietary_models = ["GPT-4V", "Gemini-Pro-V"]
    open_df = es_df[~es_df["model"].isin(proprietary_models)].copy()

    # Full meta-analysis on open-source only
    full_ma = random_effects_meta_analysis(es_df["d"].values, es_df["var"].values)
    open_ma = random_effects_meta_analysis(open_df["d"].values, open_df["var"].values)

    # Meta-regression on open-source only
    open_mr = meta_regression(open_df, "log_params")

    # Subgroup by training (should now be IT vs PT only)
    open_sub = subgroup_meta_analysis(open_df, "training_strategy")

    return {
        "n_excluded": len(es_df) - len(open_df),
        "excluded_models": proprietary_models,
        "full_analysis": {
            "k": full_ma["k"],
            "pooled_d": full_ma["pooled_effect"],
            "I_sq": full_ma["I_sq"],
            "tau_sq": full_ma["tau_sq"],
        },
        "open_source_analysis": {
            "k": open_ma["k"],
            "pooled_d": open_ma["pooled_effect"],
            "pooled_se": open_ma["pooled_se"],
            "ci_lower": open_ma["pooled_ci_lower"],
            "ci_upper": open_ma["pooled_ci_upper"],
            "I_sq": open_ma["I_sq"],
            "tau_sq": open_ma["tau_sq"],
        },
        "open_meta_regression": {
            "slope": open_mr["slope"],
            "slope_se": open_mr["slope_se"],
            "slope_p": open_mr["slope_p"],
            "R_sq": open_mr["R_sq"],
        },
        "open_subgroups": open_sub,
        "delta_pooled_d": full_ma["pooled_effect"] - open_ma["pooled_effect"],
        "delta_I_sq": full_ma["I_sq"] - open_ma["I_sq"],
    }


# ============================================================================
# 5. Galbraith (Radial) Plot Data
# ============================================================================

def galbraith_plot_data(
    effect_sizes: np.ndarray,
    standard_errors: np.ndarray,
    model_names: List[str],
    pooled_effect: float
) -> pd.DataFrame:
    """
    Compute data for a Galbraith (radial) plot.

    Plots z_i = d_i / SE_i against 1/SE_i. Points should cluster
    around the regression line through the origin if heterogeneity is low.

    Parameters
    ----------
    effect_sizes : array-like
        Model effect sizes (d_i).
    standard_errors : array-like
        Standard errors (SE_i).
    model_names : list
        Model name labels.
    pooled_effect : float
        Pooled effect from meta-analysis.

    Returns
    -------
    pd.DataFrame
        Columns: model, precision (1/SE), z_score (d/SE), expected_z.
    """
    d = np.asarray(effect_sizes, dtype=float)
    se = np.asarray(standard_errors, dtype=float)

    precision = 1.0 / se
    z_score = d / se
    expected_z = pooled_effect * precision

    return pd.DataFrame({
        "model": model_names,
        "precision": precision,
        "z_score": z_score,
        "expected_z": expected_z,
        "residual": z_score - expected_z,
        "outlier": np.abs(z_score - expected_z) > 2.0,
    })


# ============================================================================
# 6. Influence Diagnostics
# ============================================================================

def influence_diagnostics(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    model_names: List[str]
) -> pd.DataFrame:
    """
    Compute influence diagnostics for each model.

    Calculates:
    - DFBETAS (change in pooled estimate / SE when model removed)
    - Cook's distance analog
    - Standardized residual
    - Hat value (leverage)

    Parameters
    ----------
    effect_sizes : array-like
        Model effect sizes.
    variances : array-like
        Model variances.
    model_names : list
        Model name labels.

    Returns
    -------
    pd.DataFrame
        Influence metrics per model.
    """
    from statistical_analysis import random_effects_meta_analysis

    d = np.asarray(effect_sizes, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = len(d)

    full_ma = random_effects_meta_analysis(d, v)
    tau_sq = full_ma["tau_sq"]
    pooled_d = full_ma["pooled_effect"]
    pooled_se = full_ma["pooled_se"]

    # Random-effects weights
    w_re = 1.0 / (v + tau_sq)
    w_total = np.sum(w_re)

    # Hat values (leverage)
    hat_values = w_re / w_total

    # Standardized residuals
    residuals = d - pooled_d
    std_resid = residuals / np.sqrt(v + tau_sq)

    results = []
    for i in range(k):
        # Leave-one-out
        d_loo = np.delete(d, i)
        v_loo = np.delete(v, i)
        ma_loo = random_effects_meta_analysis(d_loo, v_loo)

        # DFBETAS
        dfbetas = (pooled_d - ma_loo["pooled_effect"]) / pooled_se

        # Cook's distance analog
        cooks_d = (hat_values[i] / (1 - hat_values[i])) * std_resid[i] ** 2

        results.append({
            "model": model_names[i],
            "d": d[i],
            "hat_value": hat_values[i],
            "std_residual": std_resid[i],
            "dfbetas": dfbetas,
            "cooks_distance": cooks_d,
            "influential": abs(dfbetas) > 2.0 / np.sqrt(k),
        })

    return pd.DataFrame(results)


# ============================================================================
# 7. Efficiency Analysis & Pareto Frontier
# ============================================================================

def get_efficiency_data() -> pd.DataFrame:
    """
    Return estimated computational efficiency metrics for each model.

    Estimates are based on published inference benchmarks, reported
    FLOPs, and parameter-based scaling estimates where exact data
    is unavailable (clearly marked).

    Returns
    -------
    pd.DataFrame
        Model efficiency data with TFLOPs, latency estimates.
    """
    # Estimated TFLOPs per forward pass (approximate, based on published reports)
    # For proprietary models, estimates are extrapolated from parameter counts
    records = [
        {"model": "Fuyu-8B", "params_b": 8.0, "est_tflops": 16.0,
         "est_latency_ms": 85, "open_source": True, "flops_source": "estimated"},
        {"model": "MiniGPT-4", "params_b": 8.0, "est_tflops": 18.0,
         "est_latency_ms": 95, "open_source": True, "flops_source": "estimated"},
        {"model": "BLIP-2", "params_b": 12.1, "est_tflops": 24.2,
         "est_latency_ms": 120, "open_source": True, "flops_source": "paper"},
        {"model": "InstructBLIP-7B", "params_b": 7.0, "est_tflops": 14.0,
         "est_latency_ms": 78, "open_source": True, "flops_source": "estimated"},
        {"model": "InstructBLIP-13B", "params_b": 13.0, "est_tflops": 26.0,
         "est_latency_ms": 135, "open_source": True, "flops_source": "estimated"},
        {"model": "LLaVA-1.5-7B", "params_b": 7.0, "est_tflops": 14.0,
         "est_latency_ms": 72, "open_source": True, "flops_source": "paper"},
        {"model": "LLaVA-1.5-13B", "params_b": 13.0, "est_tflops": 26.0,
         "est_latency_ms": 130, "open_source": True, "flops_source": "paper"},
        {"model": "LLaVA-NeXT-7B", "params_b": 7.0, "est_tflops": 15.5,
         "est_latency_ms": 80, "open_source": True, "flops_source": "estimated"},
        {"model": "LLaVA-NeXT-13B", "params_b": 13.0, "est_tflops": 28.5,
         "est_latency_ms": 140, "open_source": True, "flops_source": "estimated"},
        {"model": "Qwen-VL-Chat", "params_b": 9.6, "est_tflops": 19.2,
         "est_latency_ms": 98, "open_source": True, "flops_source": "estimated"},
        {"model": "mPLUG-Owl2", "params_b": 8.2, "est_tflops": 16.4,
         "est_latency_ms": 88, "open_source": True, "flops_source": "estimated"},
        {"model": "ShareGPT4V-7B", "params_b": 7.0, "est_tflops": 14.0,
         "est_latency_ms": 75, "open_source": True, "flops_source": "estimated"},
        {"model": "Yi-VL-6B", "params_b": 6.0, "est_tflops": 12.0,
         "est_latency_ms": 65, "open_source": True, "flops_source": "estimated"},
        {"model": "Yi-VL-34B", "params_b": 34.0, "est_tflops": 68.0,
         "est_latency_ms": 280, "open_source": True, "flops_source": "estimated"},
        {"model": "CogVLM-17B", "params_b": 17.0, "est_tflops": 34.0,
         "est_latency_ms": 165, "open_source": True, "flops_source": "paper"},
        {"model": "Monkey", "params_b": 9.8, "est_tflops": 22.0,
         "est_latency_ms": 105, "open_source": True, "flops_source": "estimated"},
        {"model": "Cambrian-1-8B", "params_b": 8.0, "est_tflops": 20.0,
         "est_latency_ms": 95, "open_source": True, "flops_source": "estimated"},
        {"model": "DeepSeek-VL-7B", "params_b": 7.3, "est_tflops": 14.6,
         "est_latency_ms": 76, "open_source": True, "flops_source": "estimated"},
        {"model": "InternVL-Chat-V1.5", "params_b": 26.0, "est_tflops": 52.0,
         "est_latency_ms": 210, "open_source": True, "flops_source": "paper"},
        {"model": "GPT-4V", "params_b": 1800.0, "est_tflops": 3600.0,
         "est_latency_ms": 2800, "open_source": False, "flops_source": "extrapolated"},
        {"model": "Gemini-Pro-V", "params_b": 500.0, "est_tflops": 1000.0,
         "est_latency_ms": 1500, "open_source": False, "flops_source": "extrapolated"},
    ]

    return pd.DataFrame(records)


def compute_pareto_frontier(
    es_df: pd.DataFrame,
    efficiency_df: pd.DataFrame
) -> Dict:
    """
    Compute the Pareto frontier of accuracy vs. computational efficiency.

    A model is Pareto-optimal if no other model achieves both higher
    accuracy AND lower computational cost.

    Parameters
    ----------
    es_df : pd.DataFrame
        Effect size DataFrame with model performance.
    efficiency_df : pd.DataFrame
        Efficiency data with TFLOPs estimates.

    Returns
    -------
    dict
        Merged data with Pareto labels, and the EEP (Efficiency-Efficacy
        Product) score for each model.
    """
    merged = es_df.merge(efficiency_df, on="model", how="inner", suffixes=("", "_eff"))

    # Compute EEP: normalized_performance / log(TFLOPs)
    # Higher is better — models that achieve more per unit of compute
    d_min, d_max = merged["d"].min(), merged["d"].max()
    if d_max > d_min:
        merged["norm_performance"] = (merged["d"] - d_min) / (d_max - d_min)
    else:
        merged["norm_performance"] = 0.5

    merged["log_tflops"] = np.log10(merged["est_tflops"])
    merged["eep_score"] = merged["norm_performance"] / merged["log_tflops"]

    # Identify Pareto-optimal models (maximize performance, minimize TFLOPs)
    pareto_mask = np.ones(len(merged), dtype=bool)
    for i in range(len(merged)):
        for j in range(len(merged)):
            if i == j:
                continue
            # j dominates i if j has higher performance AND lower TFLOPs
            if (merged.iloc[j]["d"] >= merged.iloc[i]["d"] and
                merged.iloc[j]["est_tflops"] <= merged.iloc[i]["est_tflops"] and
                (merged.iloc[j]["d"] > merged.iloc[i]["d"] or
                 merged.iloc[j]["est_tflops"] < merged.iloc[i]["est_tflops"])):
                pareto_mask[i] = False
                break

    merged["pareto_optimal"] = pareto_mask

    return {
        "data": merged,
        "pareto_models": merged[merged["pareto_optimal"]]["model"].tolist(),
        "n_pareto": int(pareto_mask.sum()),
        "best_eep": merged.loc[merged["eep_score"].idxmax(), "model"],
    }


# ============================================================================
# 8. Run All Robustness Analyses
# ============================================================================

def run_robustness_analyses(results: Dict, df_norm: pd.DataFrame) -> Dict:
    """
    Execute all robustness and sensitivity analyses.

    Parameters
    ----------
    results : dict
        Results from run_full_analysis().
    df_norm : pd.DataFrame
        Normalized benchmark data.

    Returns
    -------
    dict
        All robustness analysis results.
    """
    es_df = results["effect_sizes"]
    overall_ma = results["overall_meta_analysis"]

    robustness = {}

    # 1. Hedges' g
    print("\n  [R1] Computing Hedges' g effect sizes...")
    robustness["hedges_g"] = compute_all_hedges_g(df_norm)

    # 2. Leave-one-out
    print("  [R2] Leave-one-out sensitivity analysis...")
    robustness["leave_one_out"] = leave_one_out_analysis(
        es_df["d"].values, es_df["var"].values, es_df["model"].tolist()
    )

    # 3. Bootstrap correlations
    print("  [R3] Bootstrap CIs for benchmark correlations (10,000 resamples)...")
    robustness["bootstrap_correlations"] = bootstrap_correlation_ci(
        df_norm, n_bootstrap=10000
    )

    # 4. Sensitivity excluding proprietary
    print("  [R4] Sensitivity analysis excluding proprietary models...")
    robustness["sensitivity_proprietary"] = sensitivity_excluding_proprietary(es_df)

    # 5. Galbraith plot data
    print("  [R5] Computing Galbraith (radial) plot data...")
    robustness["galbraith"] = galbraith_plot_data(
        es_df["d"].values, es_df["se"].values,
        es_df["model"].tolist(), overall_ma["pooled_effect"]
    )

    # 6. Influence diagnostics
    print("  [R6] Computing influence diagnostics...")
    robustness["influence"] = influence_diagnostics(
        es_df["d"].values, es_df["var"].values, es_df["model"].tolist()
    )

    # 7. Efficiency / Pareto analysis
    print("  [R7] Computing efficiency Pareto frontier...")
    eff_df = get_efficiency_data()
    robustness["efficiency"] = eff_df
    robustness["pareto"] = compute_pareto_frontier(es_df, eff_df)

    # 8. Leave-one-benchmark-out for primary multilevel model
    print("  [R8] Leave-one-benchmark-out (primary multilevel) sensitivity...")
    robustness["benchmark_leave_one_out"] = leave_one_benchmark_out_multilevel(df_norm)

    return robustness
