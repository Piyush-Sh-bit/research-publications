# Author: Piyush Sharma
"""
multilevel_analysis.py
=======================
Multilevel mixed-effects meta-analysis for benchmark-within-model structure.

This module implements the primary analysis model that properly accounts for
the nested data structure: each model (level 2) is evaluated on multiple
benchmarks (level 1), producing non-independent observations.

Implements:
  1. Two-level mixed-effects model via REML
  2. ICC and variance component decomposition
  3. Fixed-effects moderator table with proper uncertainty
  4. SE floor sensitivity analysis
  5. Model comparison diagnostics (AIC, BIC, LRT)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# 1. Multilevel Mixed-Effects Model
# ============================================================================

def fit_multilevel_model(
    df: pd.DataFrame,
    formula: str = "zscore ~ 1",
    group_var: str = "model",
    reml: bool = True
) -> Dict:
    """
    Fit a two-level mixed-effects model: observations (benchmarks)
    nested within models.

    Model: z_ij = X_ij * beta + u_i + e_ij
      where u_i ~ N(0, sigma2_between), e_ij ~ N(0, sigma2_within)

    Parameters
    ----------
    df : pd.DataFrame
        Observation-level data with 'zscore', 'model', and predictor columns.
    formula : str
        Patsy formula for fixed effects (e.g., "zscore ~ log_params").
    group_var : str
        Column defining the grouping (model).
    reml : bool
        Use REML estimation (True) or ML (False).

    Returns
    -------
    dict
        Model results including fixed effects, variance components, ICC,
        AIC, BIC, and the fitted model object.
    """
    # Fit the mixed model
    md = smf.mixedlm(formula, df, groups=df[group_var], re_formula="~1")
    result = md.fit(reml=reml, method="lbfgs", maxiter=500)

    # Extract variance components
    sigma2_between = float(result.cov_re.iloc[0, 0])  # random intercept variance
    sigma2_within = float(result.scale)  # residual variance

    # Intra-class correlation
    icc = sigma2_between / (sigma2_between + sigma2_within) if (sigma2_between + sigma2_within) > 0 else 0.0

    # Fixed effects table
    fe_table = []
    for name in result.fe_params.index:
        coef = result.fe_params[name]
        se = result.bse_fe[name]
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        z_val = result.tvalues[name]
        p_val = result.pvalues[name]
        fe_table.append({
            "parameter": name,
            "coefficient": coef,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "z": z_val,
            "p": p_val,
        })
    fe_df = pd.DataFrame(fe_table)

    # Model fit statistics
    n_obs = int(result.nobs)
    n_groups = int(md.n_groups)

    # AIC/BIC (use log-likelihood)
    ll = result.llf
    n_params = len(result.fe_params) + 2  # fixed effects + sigma2_between + sigma2_within
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + np.log(n_obs) * n_params

    return {
        "model_object": result,
        "fixed_effects": fe_df,
        "sigma2_between": sigma2_between,
        "sigma2_within": sigma2_within,
        "icc": icc,
        "n_obs": n_obs,
        "n_groups": n_groups,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "converged": result.converged,
        "summary": str(result.summary()),
    }


# ============================================================================
# 2. Full Moderator Analysis
# ============================================================================

def fit_moderator_models(df: pd.DataFrame) -> Dict:
    """
    Fit a series of multilevel models with different moderator specifications
    to build the moderator table.

    Models fitted:
      M0: Intercept-only (null)
      M1: log(params) as continuous moderator
      M2: Training strategy as categorical moderator
      M3: Encoder family as categorical moderator
      M4: Full model (log_params + training_strategy)

    Parameters
    ----------
    df : pd.DataFrame
        Observation-level data.

    Returns
    -------
    dict
        Results for each model specification.
    """
    results = {}

    # Ensure categorical variables are properly encoded
    df = df.copy()
    df["training_cat"] = pd.Categorical(df["training_strategy"])
    df["encoder_cat"] = pd.Categorical(df["encoder_family"])

    # M0: Null model (intercept only)
    print("    M0: Intercept-only model...")
    results["M0_null"] = fit_multilevel_model(df, "zscore ~ 1", "model")

    # M1: Scale moderator
    print("    M1: Scale moderator (log_params)...")
    results["M1_scale"] = fit_multilevel_model(df, "zscore ~ log_params", "model")

    # M2: Training strategy moderator
    print("    M2: Training strategy moderator...")
    results["M2_training"] = fit_multilevel_model(
        df, "zscore ~ C(training_strategy, Treatment(reference='instruction_tuning'))", "model"
    )

    # M3: Encoder family moderator
    # Only include models with known encoder families
    encoder_df = df[df["encoder_family"].isin(["CLIP-family", "EVA-family", "ViT-large"])].copy()
    if len(encoder_df) > 10:
        print("    M3: Encoder family moderator...")
        results["M3_encoder"] = fit_multilevel_model(
            encoder_df, "zscore ~ C(encoder_family, Treatment(reference='CLIP-family'))", "model"
        )

    # M4: Full model (scale + training)
    print("    M4: Full model (scale + training)...")
    results["M4_full"] = fit_multilevel_model(
        df, "zscore ~ log_params + C(training_strategy, Treatment(reference='instruction_tuning'))", "model"
    )

    return results


def build_moderator_table(moderator_results: Dict) -> pd.DataFrame:
    """
    Build a consolidated moderator table from multilevel model results.

    Parameters
    ----------
    moderator_results : dict
        Results from fit_moderator_models().

    Returns
    -------
    pd.DataFrame
        Moderator table with coefficient, SE, CI, p, model specification.
    """
    rows = []
    for model_name, res in moderator_results.items():
        fe = res["fixed_effects"]
        for _, row in fe.iterrows():
            if row["parameter"] == "Intercept":
                # For null model, report intercept
                if model_name == "M0_null":
                    rows.append({
                        "model": model_name,
                        "parameter": "Grand Mean",
                        "coefficient": row["coefficient"],
                        "se": row["se"],
                        "ci_lower": row["ci_lower"],
                        "ci_upper": row["ci_upper"],
                        "p": row["p"],
                        "aic": res["aic"],
                        "bic": res["bic"],
                        "icc": res["icc"],
                    })
            else:
                # Clean parameter name
                param = row["parameter"]
                param = param.replace("C(training_strategy, Treatment(reference='instruction_tuning'))", "Training: ")
                param = param.replace("C(encoder_family, Treatment(reference='CLIP-family'))", "Encoder: ")
                param = param.replace("[T.", "").rstrip("]")
                rows.append({
                    "model": model_name,
                    "parameter": param,
                    "coefficient": row["coefficient"],
                    "se": row["se"],
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                    "p": row["p"],
                    "aic": res["aic"],
                    "bic": res["bic"],
                    "icc": res["icc"],
                })

    return pd.DataFrame(rows)


# ============================================================================
# 3. SE Floor Sensitivity Analysis
# ============================================================================

def se_floor_sensitivity(
    df: pd.DataFrame,
    floors: List[float] = None
) -> pd.DataFrame:
    """
    Re-run the aggregated DL meta-analysis across multiple SE floor values
    to demonstrate that conclusions are invariant to the choice of floor.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized data with 'model', 'zscore' columns.
    floors : list of float
        SE floor values to test.

    Returns
    -------
    pd.DataFrame
        Results for each floor: pooled_d, SE, CI, I², tau², meta-regression slope/p.
    """
    from statistical_analysis import random_effects_meta_analysis, meta_regression

    if floors is None:
        floors = [0.01, 0.025, 0.05, 0.10, 0.15]

    results = []
    for floor_val in floors:
        # Recompute effect sizes with this floor
        es_list = []
        for model, group in df.groupby("model"):
            zscores = group["zscore"].values
            n = len(zscores)
            d = np.mean(zscores)

            if n > 1:
                se = np.std(zscores, ddof=1) / np.sqrt(n)
            else:
                se = 0.3

            se = max(se, floor_val)

            meta = group.iloc[0]
            es_list.append({
                "model": model,
                "d": d,
                "se": se,
                "var": se ** 2,
                "log_params": meta["log_params"],
                "training_strategy": meta["training_strategy"],
            })

        es_df = pd.DataFrame(es_list)

        # Overall meta-analysis
        ma = random_effects_meta_analysis(es_df["d"].values, es_df["var"].values)

        # Meta-regression
        mr = meta_regression(es_df, "log_params")

        results.append({
            "se_floor": floor_val,
            "pooled_d": ma["pooled_effect"],
            "pooled_se": ma["pooled_se"],
            "ci_lower": ma["pooled_ci_lower"],
            "ci_upper": ma["pooled_ci_upper"],
            "I_sq": ma["I_sq"],
            "tau_sq": ma["tau_sq"],
            "Q": ma["Q"],
            "mr_slope": mr["slope"],
            "mr_slope_se": mr["slope_se"],
            "mr_slope_p": mr["slope_p"],
            "mr_R_sq": mr["R_sq"],
        })

    return pd.DataFrame(results)


# ============================================================================
# 4. Likelihood Ratio Test
# ============================================================================

def likelihood_ratio_test(ll_null: float, ll_full: float, df_diff: int) -> Dict:
    """
    Likelihood ratio test comparing nested models.

    Parameters
    ----------
    ll_null : float
        Log-likelihood of the null (simpler) model.
    ll_full : float
        Log-likelihood of the full (more complex) model.
    df_diff : int
        Difference in number of parameters.

    Returns
    -------
    dict
        chi2, df, p-value
    """
    chi2 = 2 * (ll_full - ll_null)
    chi2 = max(chi2, 0)
    p_val = 1 - stats.chi2.cdf(chi2, df_diff)

    return {
        "chi2": chi2,
        "df": df_diff,
        "p": p_val,
    }


# ============================================================================
# 5. Variance Component Decomposition Table
# ============================================================================

def variance_decomposition_table(moderator_results: Dict) -> pd.DataFrame:
    """
    Create a table showing how variance components change across model
    specifications, enabling proper heterogeneity decomposition.

    Parameters
    ----------
    moderator_results : dict
        Results from fit_moderator_models().

    Returns
    -------
    pd.DataFrame
        Variance decomposition across specifications.
    """
    rows = []
    null_sigma2_between = moderator_results.get("M0_null", {}).get("sigma2_between", 1.0)

    for model_name, res in moderator_results.items():
        # Proportion of between-model variance explained relative to null
        if null_sigma2_between > 0:
            r_sq_between = max(0, 1 - res["sigma2_between"] / null_sigma2_between)
        else:
            r_sq_between = 0.0

        rows.append({
            "model": model_name,
            "sigma2_between": res["sigma2_between"],
            "sigma2_within": res["sigma2_within"],
            "icc": res["icc"],
            "r_sq_between": r_sq_between,
            "aic": res["aic"],
            "bic": res["bic"],
            "ll": res["log_likelihood"],
            "n_obs": res["n_obs"],
            "n_groups": res["n_groups"],
        })

    return pd.DataFrame(rows)


# ============================================================================
# 6. Run All Multilevel Analyses
# ============================================================================

def run_multilevel_analyses(df_norm: pd.DataFrame) -> Dict:
    """
    Execute all multilevel analyses.

    Parameters
    ----------
    df_norm : pd.DataFrame
        Normalized benchmark data with all columns.

    Returns
    -------
    dict
        All multilevel analysis results.
    """
    print("\n  [ML] Fitting multilevel mixed-effects models...")

    multilevel = {}

    # 1. Fit moderator models
    print("  [ML.1] Fitting moderator model specifications...")
    moderator_results = fit_moderator_models(df_norm)
    multilevel["moderator_results"] = moderator_results

    # 2. Build moderator table
    print("  [ML.2] Building moderator table...")
    multilevel["moderator_table"] = build_moderator_table(moderator_results)

    # 3. Variance decomposition
    print("  [ML.3] Variance decomposition table...")
    multilevel["variance_decomposition"] = variance_decomposition_table(moderator_results)

    # 4. Likelihood ratio tests
    print("  [ML.4] Likelihood ratio tests...")
    lrt_results = {}

    null_ll = moderator_results["M0_null"]["log_likelihood"]

    for mname in ["M1_scale", "M2_training", "M4_full"]:
        if mname in moderator_results:
            n_extra = len(moderator_results[mname]["fixed_effects"]) - 1  # minus intercept
            lrt = likelihood_ratio_test(null_ll, moderator_results[mname]["log_likelihood"], max(n_extra, 1))
            lrt_results[f"M0_vs_{mname}"] = lrt

    if "M1_scale" in moderator_results and "M4_full" in moderator_results:
        lrt = likelihood_ratio_test(
            moderator_results["M1_scale"]["log_likelihood"],
            moderator_results["M4_full"]["log_likelihood"],
            max(len(moderator_results["M4_full"]["fixed_effects"]) - len(moderator_results["M1_scale"]["fixed_effects"]), 1)
        )
        lrt_results["M1_vs_M4"] = lrt

    multilevel["lrt"] = lrt_results

    # 5. SE floor sensitivity
    print("  [ML.5] SE floor sensitivity analysis...")
    multilevel["se_sensitivity"] = se_floor_sensitivity(df_norm)

    # 6. Primary model summary
    primary = moderator_results["M4_full"]
    multilevel["primary_model"] = {
        "sigma2_between": primary["sigma2_between"],
        "sigma2_within": primary["sigma2_within"],
        "icc": primary["icc"],
        "aic": primary["aic"],
        "bic": primary["bic"],
        "converged": primary["converged"],
    }

    return multilevel
