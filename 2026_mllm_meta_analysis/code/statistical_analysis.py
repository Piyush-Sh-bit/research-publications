# Author: Piyush Sharma
"""
statistical_analysis.py
========================
Core statistical methods for the MLLM meta-analysis.

Implements:
  1. Z-score normalization of benchmark scores
  2. Random-effects meta-analysis (DerSimonian-Laird)
  3. Heterogeneity statistics (I², Q, tau²)
  4. Moderator analysis (meta-regression)
  5. Publication bias tests (Egger's, funnel plot asymmetry)
  6. Subgroup analysis by architecture factors
  7. Correlation and regression analysis (scale vs performance)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# 1. Score Normalization
# ============================================================================

def normalize_scores(df: pd.DataFrame, benchmark_meta: dict) -> pd.DataFrame:
    """
    Normalize all benchmark scores to z-scores within each benchmark.
    
    For benchmarks on percentage scale, scores are first divided by max_score.
    For raw-score benchmarks (like MME), scores are divided by their max_score.
    Then z-score normalization is applied per benchmark.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw benchmark data with 'benchmark' and 'score' columns.
    benchmark_meta : dict
        Metadata dict from data_collection.get_benchmark_metadata().
    
    Returns
    -------
    pd.DataFrame
        Copy of df with added 'normalized_score' and 'zscore' columns.
    """
    df = df.copy()
    
    # Normalize to [0, 1] scale first
    df["normalized_score"] = df.apply(
        lambda row: row["score"] / benchmark_meta[row["benchmark"]]["max_score"],
        axis=1
    )
    
    # Z-score within each benchmark
    df["zscore"] = df.groupby("benchmark")["normalized_score"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )
    
    return df


# ============================================================================
# 2. Effect Size Computation
# ============================================================================

def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standardized effect sizes for each model across benchmarks.
    
    Uses normalized scores to compute a per-model mean effect size (d)
    and its standard error (SE). The effect size here is the mean z-score
    across all benchmarks a model was evaluated on.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'model' and 'zscore' columns.
    
    Returns
    -------
    pd.DataFrame
        One row per model with columns: model, d (effect size), se, 
        var, ci_lower, ci_upper, n_benchmarks, and model metadata.
    """
    results = []
    for model, group in df.groupby("model"):
        zscores = group["zscore"].values
        n = len(zscores)
        d = np.mean(zscores)
        
        # Standard error: SD / sqrt(n), with minimum floor for stability
        if n > 1:
            se = np.std(zscores, ddof=1) / np.sqrt(n)
        else:
            # Single benchmark: use pooled SD estimate
            se = 0.3  # conservative estimate
        
        se = max(se, 0.05)  # floor to prevent division by zero
        
        meta = group.iloc[0]
        results.append({
            "model": model,
            "d": d,
            "se": se,
            "var": se ** 2,
            "ci_lower": d - 1.96 * se,
            "ci_upper": d + 1.96 * se,
            "n_benchmarks": n,
            "params_b": meta["params_b"],
            "vision_encoder": meta["vision_encoder"],
            "encoder_family": meta["encoder_family"],
            "llm_backbone": meta["llm_backbone"],
            "training_strategy": meta["training_strategy"],
            "scale_category": meta["scale_category"],
            "log_params": meta["log_params"],
        })
    
    return pd.DataFrame(results).sort_values("d", ascending=True).reset_index(drop=True)


# ============================================================================
# 3. Random-Effects Meta-Analysis (DerSimonian-Laird)
# ============================================================================

def random_effects_meta_analysis(
    effect_sizes: np.ndarray, 
    variances: np.ndarray
) -> Dict:
    """
    Perform DerSimonian-Laird random-effects meta-analysis.
    
    Parameters
    ----------
    effect_sizes : array-like
        Individual study effect sizes (d_i).
    variances : array-like
        Within-study variances (v_i).
    
    Returns
    -------
    dict
        Keys: pooled_effect, pooled_se, pooled_ci_lower, pooled_ci_upper,
              pooled_z, pooled_p, tau_sq, Q, Q_p, I_sq, k
    """
    k = len(effect_sizes)
    d = np.asarray(effect_sizes, dtype=float)
    v = np.asarray(variances, dtype=float)
    
    # Fixed-effect weights
    w = 1.0 / v
    
    # Fixed-effect pooled estimate
    d_fe = np.sum(w * d) / np.sum(w)
    
    # Cochran's Q statistic
    Q = np.sum(w * (d - d_fe) ** 2)
    Q_df = k - 1
    Q_p = 1.0 - stats.chi2.cdf(Q, Q_df) if Q_df > 0 else 1.0
    
    # DerSimonian-Laird tau-squared estimator
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau_sq = max(0, (Q - Q_df) / C) if C > 0 else 0.0
    
    # Random-effects weights
    w_re = 1.0 / (v + tau_sq)
    
    # Pooled random-effects estimate
    d_re = np.sum(w_re * d) / np.sum(w_re)
    se_re = np.sqrt(1.0 / np.sum(w_re))
    
    # Confidence interval and test
    ci_lower = d_re - 1.96 * se_re
    ci_upper = d_re + 1.96 * se_re
    z_val = d_re / se_re
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_val)))
    
    # I-squared
    I_sq = max(0, (Q - Q_df) / Q * 100) if Q > 0 else 0.0
    
    return {
        "pooled_effect": d_re,
        "pooled_se": se_re,
        "pooled_ci_lower": ci_lower,
        "pooled_ci_upper": ci_upper,
        "pooled_z": z_val,
        "pooled_p": p_val,
        "tau_sq": tau_sq,
        "Q": Q,
        "Q_df": Q_df,
        "Q_p": Q_p,
        "I_sq": I_sq,
        "k": k,
    }


def subgroup_meta_analysis(
    es_df: pd.DataFrame, 
    groupby_col: str
) -> pd.DataFrame:
    """
    Perform subgroup meta-analysis by a categorical moderator.
    
    Parameters
    ----------
    es_df : pd.DataFrame
        Effect size DataFrame from compute_effect_sizes().
    groupby_col : str
        Column name to group by.
    
    Returns
    -------
    pd.DataFrame
        One row per subgroup with meta-analysis results.
    """
    results = []
    for group_name, sub in es_df.groupby(groupby_col):
        if len(sub) < 2:
            continue
        ma = random_effects_meta_analysis(sub["d"].values, sub["var"].values)
        ma["subgroup"] = group_name
        ma["n_models"] = len(sub)
        results.append(ma)
    
    return pd.DataFrame(results)


# ============================================================================
# 4. Meta-Regression
# ============================================================================

def meta_regression(
    es_df: pd.DataFrame, 
    predictor: str = "log_params"
) -> Dict:
    """
    Mixed-effects meta-regression with a continuous moderator.
    
    Uses weighted least squares with random-effects weights.
    
    Parameters
    ----------
    es_df : pd.DataFrame
        Effect size DataFrame.
    predictor : str
        Column name of the continuous predictor.
    
    Returns
    -------
    dict
        Regression results including coefficients, p-values, R².
    """
    d = es_df["d"].values
    v = es_df["var"].values
    x = es_df[predictor].values
    
    # First compute tau_sq from overall meta-analysis
    overall = random_effects_meta_analysis(d, v)
    tau_sq = overall["tau_sq"]
    
    # Random-effects weights for WLS
    w = 1.0 / (v + tau_sq)
    
    # Weighted least squares
    X = sm.add_constant(x)
    wls = sm.WLS(d, X, weights=w).fit()
    
    # Compute QM (moderator Q) and QE (residual Q)
    d_pred = wls.predict(X)
    QE = np.sum(w * (d - d_pred) ** 2)
    QM = np.sum(w * (d_pred - np.average(d, weights=w)) ** 2)
    QE_df = len(d) - 2
    QM_df = 1
    QM_p = 1.0 - stats.chi2.cdf(QM, QM_df) if QM > 0 else 1.0
    
    # Proportion of heterogeneity explained
    overall_QE = overall["Q"]
    R_sq = max(0, 1.0 - QE / overall_QE) if overall_QE > 0 else 0.0
    
    return {
        "intercept": wls.params[0],
        "slope": wls.params[1],
        "intercept_se": wls.bse[0],
        "slope_se": wls.bse[1],
        "intercept_p": wls.pvalues[0],
        "slope_p": wls.pvalues[1],
        "R_sq": R_sq,
        "QM": QM,
        "QM_df": QM_df,
        "QM_p": QM_p,
        "QE": QE,
        "QE_df": QE_df,
        "n": len(d),
        "predictor": predictor,
        "wls_summary": str(wls.summary()),
    }


# ============================================================================
# 5. Publication Bias Tests
# ============================================================================

def eggers_test(
    effect_sizes: np.ndarray, 
    standard_errors: np.ndarray
) -> Dict:
    """
    Egger's regression test for funnel plot asymmetry.
    
    Regresses standardized effect sizes (d/SE) on precision (1/SE).
    Significant intercept suggests publication bias.
    
    Parameters
    ----------
    effect_sizes : array-like
        Effect sizes (d).
    standard_errors : array-like
        Standard errors of effect sizes.
    
    Returns
    -------
    dict
        intercept, slope, intercept_se, intercept_t, intercept_p, n
    """
    d = np.asarray(effect_sizes, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    
    # Standardized effect = d_i / se_i  (dependent variable)
    y = d / se
    # Precision = 1 / se_i  (independent variable)
    x = 1.0 / se
    
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    
    return {
        "intercept": ols.params[0],
        "slope": ols.params[1],
        "intercept_se": ols.bse[0],
        "intercept_t": ols.tvalues[0],
        "intercept_p": ols.pvalues[0],
        "n": len(d),
    }


def trim_and_fill(
    effect_sizes: np.ndarray, 
    variances: np.ndarray, 
    side: str = "left"
) -> Dict:
    """
    Trim-and-fill method for estimating missing studies.
    
    Implements the L0 estimator (Duval & Tweedie, 2000).
    
    Parameters
    ----------
    effect_sizes : array-like
        Effect sizes.
    variances : array-like
        Within-study variances.
    side : str
        Side to trim ("left" or "right").
    
    Returns
    -------
    dict
        n_missing, adjusted_effect, adjusted_ci_lower, adjusted_ci_upper,
        original_effect
    """
    d = np.asarray(effect_sizes, dtype=float)
    v = np.asarray(variances, dtype=float)
    
    # Original meta-analysis
    original_ma = random_effects_meta_analysis(d, v)
    theta = original_ma["pooled_effect"]
    
    # Rank-based estimator for number of missing studies
    n = len(d)
    
    # Deviations from pooled estimate
    if side == "left":
        # Look for missing studies with small effect sizes
        deviations = theta - d
    else:
        deviations = d - theta
    
    # Sort and estimate number of missing
    ranks = stats.rankdata(np.abs(d - theta))
    
    # Simplified L0 estimator
    positive_devs = np.sum(deviations > 0)
    k0 = max(0, 2 * positive_devs - n)
    k0 = int(round(k0))
    
    # Impute missing studies
    if k0 > 0:
        # Mirror the k0 most extreme studies on the opposite side
        sorted_idx = np.argsort(d) if side == "left" else np.argsort(-d)
        mirror_idx = sorted_idx[:k0]
        
        d_fill = np.concatenate([d, 2 * theta - d[mirror_idx]])
        v_fill = np.concatenate([v, v[mirror_idx]])
        
        adjusted_ma = random_effects_meta_analysis(d_fill, v_fill)
    else:
        adjusted_ma = original_ma
    
    return {
        "n_missing": k0,
        "original_effect": original_ma["pooled_effect"],
        "original_ci_lower": original_ma["pooled_ci_lower"],
        "original_ci_upper": original_ma["pooled_ci_upper"],
        "adjusted_effect": adjusted_ma["pooled_effect"],
        "adjusted_ci_lower": adjusted_ma["pooled_ci_lower"],
        "adjusted_ci_upper": adjusted_ma["pooled_ci_upper"],
    }


# ============================================================================
# 6. Benchmark-Level Analysis
# ============================================================================

def benchmark_correlation_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise Spearman correlations between benchmarks
    (based on per-model scores).
    
    Parameters
    ----------
    df : pd.DataFrame
        Normalized data with 'model', 'benchmark', 'normalized_score'.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Correlation matrix and p-value matrix.
    """
    # Pivot to models x benchmarks
    pivot = df.pivot_table(
        index="model", columns="benchmark", values="normalized_score"
    )
    
    benchmarks = pivot.columns.tolist()
    n = len(benchmarks)
    corr_mat = pd.DataFrame(np.zeros((n, n)), index=benchmarks, columns=benchmarks)
    pval_mat = pd.DataFrame(np.ones((n, n)), index=benchmarks, columns=benchmarks)
    
    for i in range(n):
        for j in range(i, n):
            # Get models with scores on both benchmarks
            valid = pivot[[benchmarks[i], benchmarks[j]]].dropna()
            if len(valid) >= 3:
                rho, p = stats.spearmanr(valid.iloc[:,0], valid.iloc[:,1])
                corr_mat.iloc[i,j] = rho
                corr_mat.iloc[j,i] = rho
                pval_mat.iloc[i,j] = p
                pval_mat.iloc[j,i] = p
            else:
                corr_mat.iloc[i,j] = np.nan
                corr_mat.iloc[j,i] = np.nan
    
    return corr_mat, pval_mat


def scale_performance_regression(df: pd.DataFrame) -> Dict:
    """
    Analyze the relationship between model scale (log params) 
    and benchmark performance using robust regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'log_params', 'normalized_score', 'benchmark' columns.
    
    Returns
    -------
    dict
        Per-benchmark regression results + overall regression.
    """
    results = {}
    
    # Overall regression
    valid = df[["log_params", "normalized_score"]].dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        valid["log_params"], valid["normalized_score"]
    )
    results["overall"] = {
        "slope": slope,
        "intercept": intercept,
        "r": r_value,
        "r_sq": r_value ** 2,
        "p": p_value,
        "se": std_err,
        "n": len(valid),
    }
    
    # Per-benchmark regressions
    for bench, group in df.groupby("benchmark"):
        valid = group[["log_params", "normalized_score"]].dropna()
        if len(valid) >= 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid["log_params"], valid["normalized_score"]
            )
            # Also compute Spearman
            rho, rho_p = stats.spearmanr(valid["log_params"], valid["normalized_score"])
            results[bench] = {
                "slope": slope,
                "intercept": intercept,
                "r": r_value,
                "r_sq": r_value ** 2,
                "p": p_value,
                "se": std_err,
                "n": len(valid),
                "spearman_rho": rho,
                "spearman_p": rho_p,
            }
    
    return results


# ============================================================================
# 7. Kruskal-Wallis Tests for Group Comparisons
# ============================================================================

def kruskal_wallis_by_group(
    df: pd.DataFrame, 
    group_col: str, 
    value_col: str = "normalized_score"
) -> Dict:
    """
    Non-parametric Kruskal-Wallis H test for group differences.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with group and value columns.
    group_col : str
        Column defining groups.
    value_col : str
        Column with values to compare.
    
    Returns
    -------
    dict
        H statistic, p-value, effect size (eta²), group medians.
    """
    groups = []
    group_names = []
    group_medians = {}
    
    for name, sub in df.groupby(group_col):
        vals = sub[value_col].dropna().values
        if len(vals) >= 2:
            groups.append(vals)
            group_names.append(name)
            group_medians[name] = {
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    
    if len(groups) < 2:
        return {"H": np.nan, "p": np.nan, "eta_sq": np.nan, 
                "group_stats": group_medians, "valid": False}
    
    H, p = stats.kruskal(*groups)
    
    # Effect size: eta-squared for Kruskal-Wallis
    N = sum(len(g) for g in groups)
    k = len(groups)
    eta_sq = (H - k + 1) / (N - k) if N > k else 0.0
    
    return {
        "H": float(H),
        "p": float(p),
        "eta_sq": float(eta_sq),
        "group_stats": group_medians,
        "n_groups": k,
        "N": N,
        "valid": True,
    }


# ============================================================================
# 8. Comprehensive Analysis Runner
# ============================================================================

def run_full_analysis(df: pd.DataFrame, benchmark_meta: dict) -> Dict:
    """
    Execute the full meta-analysis pipeline and return all results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw benchmark data from data_collection.get_benchmark_data().
    benchmark_meta : dict
        From data_collection.get_benchmark_metadata().
    
    Returns
    -------
    dict
        Nested dictionary with all analysis results.
    """
    results = {}
    
    # Step 1: Normalize scores
    df_norm = normalize_scores(df, benchmark_meta)
    results["data"] = df_norm
    
    # Step 2: Compute effect sizes
    es_df = compute_effect_sizes(df_norm)
    results["effect_sizes"] = es_df
    
    # Step 3: Overall random-effects meta-analysis
    overall_ma = random_effects_meta_analysis(es_df["d"].values, es_df["var"].values)
    results["overall_meta_analysis"] = overall_ma
    
    # Step 4: Subgroup analyses
    results["subgroup_training"] = subgroup_meta_analysis(es_df, "training_strategy")
    results["subgroup_encoder"] = subgroup_meta_analysis(es_df, "encoder_family")
    results["subgroup_scale"] = subgroup_meta_analysis(es_df, "scale_category")
    
    # Step 5: Meta-regression (scale as moderator)
    results["meta_regression_scale"] = meta_regression(es_df, "log_params")
    
    # Step 6: Publication bias
    results["eggers_test"] = eggers_test(es_df["d"].values, es_df["se"].values)
    results["trim_and_fill"] = trim_and_fill(es_df["d"].values, es_df["var"].values)
    
    # Step 7: Benchmark correlations
    corr_mat, pval_mat = benchmark_correlation_matrix(df_norm)
    results["benchmark_correlations"] = corr_mat
    results["benchmark_corr_pvalues"] = pval_mat
    
    # Step 8: Scale-performance analysis
    results["scale_regression"] = scale_performance_regression(df_norm)
    
    # Step 9: Kruskal-Wallis tests
    results["kw_training"] = kruskal_wallis_by_group(df_norm, "training_strategy")
    results["kw_encoder"] = kruskal_wallis_by_group(df_norm, "encoder_family")
    
    return results


if __name__ == "__main__":
    from data_collection import get_benchmark_data, get_benchmark_metadata
    
    df = get_benchmark_data()
    meta = get_benchmark_metadata()
    
    results = run_full_analysis(df, meta)
    
    # Print key results
    ma = results["overall_meta_analysis"]
    print("=" * 60)
    print("OVERALL RANDOM-EFFECTS META-ANALYSIS")
    print("=" * 60)
    print(f"  Pooled effect (d): {ma['pooled_effect']:.4f}")
    print(f"  95% CI: [{ma['pooled_ci_lower']:.4f}, {ma['pooled_ci_upper']:.4f}]")
    print(f"  Z = {ma['pooled_z']:.4f}, p = {ma['pooled_p']:.4f}")
    print(f"  tau² = {ma['tau_sq']:.4f}")
    print(f"  Q({ma['Q_df']}) = {ma['Q']:.2f}, p = {ma['Q_p']:.4f}")
    print(f"  I² = {ma['I_sq']:.1f}%")
    print(f"  k = {ma['k']} models")
    
    print("\n" + "=" * 60)
    print("META-REGRESSION: log(Parameters) → Effect Size")
    print("=" * 60)
    mr = results["meta_regression_scale"]
    print(f"  Slope = {mr['slope']:.4f} (SE = {mr['slope_se']:.4f}), p = {mr['slope_p']:.4f}")
    print(f"  R² = {mr['R_sq']:.4f}")
    print(f"  QM({mr['QM_df']}) = {mr['QM']:.2f}, p = {mr['QM_p']:.4f}")
    
    print("\n" + "=" * 60)
    print("EGGER'S TEST FOR PUBLICATION BIAS")
    print("=" * 60)
    eg = results["eggers_test"]
    print(f"  Intercept = {eg['intercept']:.4f} (SE = {eg['intercept_se']:.4f})")
    print(f"  t = {eg['intercept_t']:.4f}, p = {eg['intercept_p']:.4f}")
