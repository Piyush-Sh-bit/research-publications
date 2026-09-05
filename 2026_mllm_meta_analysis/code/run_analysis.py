"""
run_analysis.py
================
Main entry point for the MLLM Meta-Analysis pipeline.

Usage:
    python run_analysis.py

This script:
  1. Loads compiled benchmark data from 21 MLLM papers
  2. Normalizes scores across 7 benchmarks
    3. Fits a multilevel mixed-effects model with model-level random intercepts
    4. Computes standardized effect sizes per model for secondary analysis
    5. Runs DerSimonian-Laird random-effects meta-analysis as a sensitivity check
    6. Performs subgroup analyses (training strategy, encoder family, scale)
    7. Conducts meta-regression (model scale -> performance)
    8. Tests for publication bias (Egger's test, trim-and-fill)
    9. Computes inter-benchmark Spearman correlations
    10. Runs robustness analyses (Hedges' g, leave-one-out, bootstrap CIs,
     sensitivity excluding proprietary, Galbraith plot, influence diagnostics,
     efficiency Pareto frontier)
    11. Generates 12 publication-quality figures
    12. Outputs comprehensive results tables (CSV + console)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collection import get_benchmark_data, get_benchmark_metadata
from statistical_analysis import (
    run_full_analysis,
    normalize_scores,
    compute_effect_sizes,
    random_effects_meta_analysis,
    subgroup_meta_analysis,
    meta_regression,
    meta_regression_multivariate,
    eggers_test,
    trim_and_fill,
    benchmark_correlation_matrix,
    scale_performance_regression,
)
from visualization import generate_all_figures
from robustness_analysis import run_robustness_analyses
from visualization_robustness import generate_robustness_figures
from multilevel_analysis import run_multilevel_analyses


def print_header(title: str, width: int = 70):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def format_p(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"= {p:.3f}"
    elif p < 0.05:
        return f"= {p:.3f}"
    else:
        return f"= {p:.3f}"


def main():
    # ================================================================
    # Setup
    # ================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    figures_dir = os.path.join(project_dir, "figures")
    tables_dir = os.path.join(project_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    print_header("MLLM VISUAL REASONING META-ANALYSIS PIPELINE")
    print(f"  Output figures: {figures_dir}")
    print(f"  Output tables:  {tables_dir}")
    
    # ================================================================
    # Step 1: Load Data
    # ================================================================
    print_header("STEP 1: DATA LOADING")
    df = get_benchmark_data()
    benchmark_meta = get_benchmark_metadata()
    
    print(f"  Total records:    {len(df)}")
    print(f"  Unique models:    {df['model'].nunique()}")
    print(f"  Unique benchmarks:{df['benchmark'].nunique()}")
    print(f"  Year range:       {df['year'].min()}-{df['year'].max()}")
    print(f"  Parameter range:  {df['params_b'].min():.1f}B - {df['params_b'].max():.0f}B")
    
    print(f"\n  Models included ({df['model'].nunique()}):")
    for m in sorted(df['model'].unique()):
        n = len(df[df['model'] == m])
        p = df[df['model'] == m]['params_b'].iloc[0]
        print(f"    - {m} ({p:.1f}B params, {n} benchmarks)")
    
    print(f"\n  Benchmarks ({df['benchmark'].nunique()}):")
    for b in sorted(df['benchmark'].unique()):
        n = len(df[df['benchmark'] == b])
        print(f"    - {b}: {n} models evaluated")
    
    # ================================================================
    # Step 2: Run Full Analysis
    # ================================================================
    print_header("STEP 2: RUNNING FULL STATISTICAL ANALYSIS")
    results = run_full_analysis(df, benchmark_meta)
    
    # ================================================================
    # Step 3: Report Secondary Aggregate Meta-Analysis
    # ================================================================
    print_header("STEP 3: SECONDARY RANDOM-EFFECTS META-ANALYSIS")
    ma = results["overall_meta_analysis"]
    
    print(f"  Method:           DerSimonian-Laird Random Effects")
    print(f"  Number of models: k = {ma['k']}")
    print(f"  Pooled effect:    d = {ma['pooled_effect']:.4f}")
    print(f"  Standard error:   SE = {ma['pooled_se']:.4f}")
    print(f"  95% CI:           [{ma['pooled_ci_lower']:.4f}, {ma['pooled_ci_upper']:.4f}]")
    print(f"  Z-test:           Z = {ma['pooled_z']:.4f}, p {format_p(ma['pooled_p'])}")
    print(f"\n  Heterogeneity:")
    print(f"    tau^2 (tau-squared): {ma['tau_sq']:.4f}")
    print(f"    Q({ma['Q_df']}):           {ma['Q']:.2f}, p {format_p(ma['Q_p'])}")
    print(f"    I^2:               {ma['I_sq']:.1f}%")
    
    if ma['I_sq'] < 25:
        hetero_level = "LOW"
    elif ma['I_sq'] < 75:
        hetero_level = "MODERATE"
    else:
        hetero_level = "HIGH"
    print(f"    Heterogeneity:    {hetero_level}")
    
    # ================================================================
    # Step 4: Subgroup Analyses
    # ================================================================
    print_header("STEP 4: SUBGROUP ANALYSES")
    
    print("\n  --- By Training Strategy ---")
    sub_train = results["subgroup_training"]
    for _, row in sub_train.iterrows():
        print(f"\n  [{row['subgroup']}] (k = {row['n_models']})")
        print(f"    Pooled d = {row['pooled_effect']:.4f} "
              f"[{row['pooled_ci_lower']:.4f}, {row['pooled_ci_upper']:.4f}]")
        print(f"    I^2 = {row['I_sq']:.1f}%, Q({row['Q_df']}) = {row['Q']:.2f}")
    
    print("\n  --- By Vision Encoder Family ---")
    sub_enc = results["subgroup_encoder"]
    for _, row in sub_enc.iterrows():
        print(f"\n  [{row['subgroup']}] (k = {row['n_models']})")
        print(f"    Pooled d = {row['pooled_effect']:.4f} "
              f"[{row['pooled_ci_lower']:.4f}, {row['pooled_ci_upper']:.4f}]")
        print(f"    I^2 = {row['I_sq']:.1f}%")
    
    print("\n  --- By Model Scale ---")
    sub_scale = results["subgroup_scale"]
    for _, row in sub_scale.iterrows():
        print(f"\n  [{row['subgroup']}] (k = {row['n_models']})")
        print(f"    Pooled d = {row['pooled_effect']:.4f} "
              f"[{row['pooled_ci_lower']:.4f}, {row['pooled_ci_upper']:.4f}]")
        print(f"    I^2 = {row['I_sq']:.1f}%")
    
    # ================================================================
    # Step 5: Meta-Regression
    # ================================================================
    print_header("STEP 5: META-REGRESSION (Scale -> Performance)")
    mr = results["meta_regression_scale"]
    print(f"  Predictor:      log10(Parameters)")
    print(f"  Intercept:      {mr['intercept']:.4f} (SE = {mr['intercept_se']:.4f}), "
          f"p {format_p(mr['intercept_p'])}")
    print(f"  Slope (beta1):     {mr['slope']:.4f} (SE = {mr['slope_se']:.4f}), "
          f"p {format_p(mr['slope_p'])}")
    print(f"  R^2:             {mr['R_sq']:.4f}")
    print(f"  QM({mr['QM_df']}):          {mr['QM']:.2f}, p {format_p(mr['QM_p'])}")
    print(f"  QE({mr['QE_df']}):          {mr['QE']:.2f}")

    # Step 5b: Multivariate meta-regression (scale + year)
    print("\n  --- Multivariate Meta-Regression (Scale + Year) ---")
    # Need year column in effect sizes df
    es_df_with_year = results["effect_sizes"].copy()
    year_map = results["data"].drop_duplicates("model").set_index("model")["year"]
    es_df_with_year["year"] = es_df_with_year["model"].map(year_map)
    mr_multi = meta_regression_multivariate(es_df_with_year, ["log_params", "year"])
    print(f"  Predictors:     log10(Parameters), Publication Year")
    print(f"  Intercept:      {mr_multi['intercept']:.4f} (SE = {mr_multi['intercept_se']:.4f}), "
          f"p {format_p(mr_multi['intercept_p'])}")
    print(f"  log_params (beta1): {mr_multi['log_params_coef']:.4f} "
          f"(SE = {mr_multi['log_params_se']:.4f}), p {format_p(mr_multi['log_params_p'])}")
    print(f"  year (beta2):      {mr_multi['year_coef']:.4f} "
          f"(SE = {mr_multi['year_se']:.4f}), p {format_p(mr_multi['year_p'])}")
    print(f"  R^2:             {mr_multi['R_sq']:.4f}")
    print(f"  QM({mr_multi['QM_df']}):          {mr_multi['QM']:.2f}, p {format_p(mr_multi['QM_p'])}")
    
    # ================================================================
    # Step 6: Publication Bias
    # ================================================================
    print_header("STEP 6: PUBLICATION BIAS ASSESSMENT")
    
    eg = results["eggers_test"]
    print(f"  Egger's Regression Test:")
    print(f"    Intercept = {eg['intercept']:.4f} (SE = {eg['intercept_se']:.4f})")
    print(f"    t = {eg['intercept_t']:.4f}, p {format_p(eg['intercept_p'])}")
    bias = "DETECTED" if eg['intercept_p'] < 0.05 else "NOT DETECTED"
    print(f"    Funnel asymmetry: {bias} (alpha = 0.05)")
    
    tf = results["trim_and_fill"]
    print(f"\n  Trim-and-Fill Analysis:")
    print(f"    Estimated missing studies: {tf['n_missing']}")
    print(f"    Original pooled d:   {tf['original_effect']:.4f} "
          f"[{tf['original_ci_lower']:.4f}, {tf['original_ci_upper']:.4f}]")
    print(f"    Adjusted pooled d:   {tf['adjusted_effect']:.4f} "
          f"[{tf['adjusted_ci_lower']:.4f}, {tf['adjusted_ci_upper']:.4f}]")
    
    # ================================================================
    # Step 7: Benchmark Correlations
    # ================================================================
    print_header("STEP 7: INTER-BENCHMARK CORRELATIONS (Spearman)")
    corr = results["benchmark_correlations"]
    print(f"\n{corr.round(3).to_string()}")
    
    # ================================================================
    # Step 8: Scale-Performance Regressions
    # ================================================================
    print_header("STEP 8: SCALE-PERFORMANCE ANALYSIS (Per Benchmark)")
    sr = results["scale_regression"]
    print(f"\n  {'Benchmark':<15} {'r':>8} {'r^2':>8} {'p':>10} {'rho':>8} {'n':>4}")
    print(f"  {'-'*55}")
    for bench, res in sorted(sr.items()):
        if bench == "overall":
            continue
        rho = res.get('spearman_rho', float('nan'))
        print(f"  {bench:<15} {res['r']:>8.3f} {res['r_sq']:>8.3f} "
              f"{res['p']:>10.4f} {rho:>8.3f} {res['n']:>4}")
    ov = sr.get("overall", {})
    print(f"  {'OVERALL':<15} {ov.get('r',0):>8.3f} {ov.get('r_sq',0):>8.3f} "
          f"{ov.get('p',1):>10.4f} {'---':>8} {ov.get('n',0):>4}")
    
    # ================================================================
    # ================================================================
    # Step 9.5: MULTILEVEL MIXED-EFFECTS ANALYSIS (PRIMARY)
    # ================================================================
    print_header("STEP 9.5: MULTILEVEL MIXED-EFFECTS ANALYSIS (PRIMARY)")
    multilevel = run_multilevel_analyses(results["data"])
    
    # Primary model summary
    pm = multilevel["primary_model"]
    print(f"\n  Primary Model (M4: scale + training):")
    print(f"    sigma2_between (model): {pm['sigma2_between']:.4f}")
    print(f"    sigma2_within (bench):  {pm['sigma2_within']:.4f}")
    print(f"    ICC:                    {pm['icc']:.4f}")
    print(f"    AIC:                    {pm['aic']:.2f}")
    print(f"    BIC:                    {pm['bic']:.2f}")
    print(f"    Converged:              {pm['converged']}")
    
    # Moderator table
    print("\n  Moderator Table (Fixed Effects):")
    mod_table = multilevel["moderator_table"]
    for _, row in mod_table.iterrows():
        print(f"    [{row['model']}] {row['parameter']}: "
              f"coef = {row['coefficient']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
              f"p {format_p(row['p'])}")
    
    # Variance decomposition
    print("\n  Variance Decomposition Across Specifications:")
    vd = multilevel["variance_decomposition"]
    for _, row in vd.iterrows():
        print(f"    {row['model']}: sigma2_b = {row['sigma2_between']:.4f}, "
              f"ICC = {row['icc']:.3f}, R2_between = {row['r_sq_between']:.3f}, "
              f"AIC = {row['aic']:.1f}")
    
    # LRT results
    print("\n  Likelihood Ratio Tests:")
    for test_name, lrt in multilevel["lrt"].items():
        print(f"    {test_name}: chi2 = {lrt['chi2']:.2f}, df = {lrt['df']}, "
              f"p {format_p(lrt['p'])}")
    
    # SE floor sensitivity
    print("\n  SE Floor Sensitivity Analysis:")
    se_sens = multilevel["se_sensitivity"]
    print(f"    {'Floor':>8} {'Pooled d':>10} {'I^2':>8} {'tau^2':>8} {'MR slope':>10} {'MR p':>10}")
    print(f"    {'-'*58}")
    for _, row in se_sens.iterrows():
        print(f"    {row['se_floor']:>8.3f} {row['pooled_d']:>10.4f} "
              f"{row['I_sq']:>8.1f} {row['tau_sq']:>8.4f} "
              f"{row['mr_slope']:>10.4f} {row['mr_slope_p']:>10.4f}")


    # Step 10: Robustness Analyses
    # ================================================================
    print_header("STEP 10: ROBUSTNESS & SENSITIVITY ANALYSES")
    robustness = run_robustness_analyses(results, results["data"])
    
    # 10a: Hedges' g comparison
    print("\n  --- Hedges' g vs. Cohen's d Comparison ---")
    hg_df = robustness["hedges_g"]
    corr_dg = np.corrcoef(hg_df["d"].values, hg_df["hedges_g"].values)[0, 1]
    max_diff = (hg_df["d"] - hg_df["hedges_g"]).abs().max()
    mean_diff = (hg_df["d"] - hg_df["hedges_g"]).abs().mean()
    print(f"    Correlation (d vs g): r = {corr_dg:.6f}")
    print(f"    Max |d - g|: {max_diff:.4f}")
    print(f"    Mean |d - g|: {mean_diff:.4f}")
    
    # 10b: Leave-one-out
    print("\n  --- Leave-One-Out Sensitivity ---")
    loo_df = robustness["leave_one_out"]
    max_influence = loo_df.iloc[0]
    print(f"    Most influential model: {max_influence['excluded_model']}")
    print(f"    Deltad when excluded: {max_influence['delta_d']:+.4f}")
    print(f"    Range of pooled d: [{loo_df['pooled_d'].min():.4f}, {loo_df['pooled_d'].max():.4f}]")
    
    # 10c: Bootstrap correlations 
    print("\n  --- Bootstrap CIs for Correlations ---")
    boot_df = robustness["bootstrap_correlations"]
    n_reliable = boot_df["reliable"].sum()
    n_total = len(boot_df)
    print(f"    Reliable pairs (n>=5, CI width<0.8): {n_reliable}/{n_total}")
    unreliable = boot_df[~boot_df["reliable"]]
    if len(unreliable) > 0:
        print(f"    Unreliable pairs:")
        for _, row in unreliable.iterrows():
            print(f"      {row['benchmark_1']}--{row['benchmark_2']}: "
                  f"n={row['n']}, rho={row.get('rho', float('nan')):.3f}")
    
    # 10d: Proprietary sensitivity
    print("\n  --- Sensitivity: Excluding Proprietary Models ---")
    sens = robustness["sensitivity_proprietary"]
    os_analysis = sens["open_source_analysis"]
    print(f"    Excluded: {', '.join(sens['excluded_models'])}")
    print(f"    Open-source pooled d: {os_analysis['pooled_d']:.4f} "
          f"[{os_analysis['ci_lower']:.4f}, {os_analysis['ci_upper']:.4f}]")
    print(f"    Delta pooled d: {sens['delta_pooled_d']:+.4f}")
    print(f"    Open-source I^2: {os_analysis['I_sq']:.1f}% "
          f"(Delta = {sens['delta_I_sq']:+.1f}%)")
    print(f"    Open-source meta-regression: beta = {sens['open_meta_regression']['slope']:.4f} "
          f"(p {format_p(sens['open_meta_regression']['slope_p'])})")
    
    # 10e: Influence diagnostics
    print("\n  --- Influence Diagnostics ---")
    inf_df = robustness["influence"]
    n_influential = inf_df["influential"].sum()
    print(f"    Influential models (|DFBETAS| > 2/sqrtk): {n_influential}")
    for _, row in inf_df[inf_df["influential"]].iterrows():
        print(f"      {row['model']}: DFBETAS = {row['dfbetas']:.4f}, "
              f"Cook's D = {row['cooks_distance']:.4f}")
    
    # 10f: Galbraith outliers
    print("\n  --- Galbraith Plot Outliers ---")
    galbraith = robustness["galbraith"]
    n_outliers = galbraith["outlier"].sum()
    print(f"    Outliers (|residual| > 2): {n_outliers}")
    for _, row in galbraith[galbraith["outlier"]].iterrows():
        print(f"      {row['model']}: residual = {row['residual']:.2f}")
    
    # 10g: Pareto frontier
    print("\n  --- Efficiency Pareto Frontier ---")
    pareto = robustness["pareto"]
    print(f"    Pareto-optimal models ({pareto['n_pareto']}):")
    for m in pareto["pareto_models"]:
        print(f"      - {m}")
    print(f"    Best Efficiency-Efficacy score: {pareto['best_eep']}")

    # 10h-extra: Open-weights-only Pareto frontier
    print("\n  --- Open-Weights Pareto Frontier (excluding proprietary) ---")
    from robustness_analysis import compute_pareto_frontier
    eff_df = robustness["efficiency"]
    eff_open = eff_df[eff_df["open_source"] == True].copy()
    es_open = results["effect_sizes"][~results["effect_sizes"]["model"].isin(["GPT-4V", "Gemini-Pro-V"])].copy()
    pareto_open = compute_pareto_frontier(es_open, eff_open)
    robustness["pareto_open_weights"] = pareto_open
    print(f"    Open-weights Pareto-optimal models ({pareto_open['n_pareto']}):")
    for m in pareto_open["pareto_models"]:
        print(f"      - {m}")
    print(f"    Best open-weights EEP: {pareto_open['best_eep']}")

    # 10h: Leave-one-benchmark-out (primary multilevel)
    print("\n  --- Leave-One-Benchmark-Out (Primary Multilevel) ---")
    bloo_df = robustness["benchmark_leave_one_out"]
    top_bloo = bloo_df.iloc[0]
    print(f"    Most influential benchmark: {top_bloo['excluded_benchmark']}")
    print(f"    Delta scale coef: {top_bloo['delta_scale_coef']:+.4f}")
    print(f"    Scale coef range: [{bloo_df['scale_coef'].min():.4f}, {bloo_df['scale_coef'].max():.4f}]")
    
    # ================================================================
    # Step 11: Export Tables
    # ================================================================
    print_header("STEP 11: EXPORTING TABLES")
    
    # Table 1: Model overview
    model_overview = df.drop_duplicates("model")[
        ["model", "params_b", "vision_encoder", "llm_backbone", 
         "training_strategy", "year"]
    ].sort_values("params_b")
    model_overview.to_csv(os.path.join(tables_dir, "table1_model_overview.csv"), index=False)
    print(f"  [OK] Table 1: Model overview ({len(model_overview)} models)")
    
    # Table 2: Raw benchmark scores
    pivot_raw = df.pivot_table(
        index="model", columns="benchmark", values="score"
    ).round(1)
    pivot_raw.to_csv(os.path.join(tables_dir, "table2_raw_scores.csv"))
    print(f"  [OK] Table 2: Raw benchmark scores")
    
    # Table 3: Effect sizes (including Hedges' g)
    es_export = results["effect_sizes"][
        ["model", "params_b", "training_strategy", "d", "se", 
         "ci_lower", "ci_upper", "n_benchmarks"]
    ].round(4)
    # Merge Hedges' g
    hg_export = hg_df[["model", "hedges_g", "se_g"]].round(4)
    es_export = es_export.merge(hg_export, on="model", how="left")
    es_export.to_csv(os.path.join(tables_dir, "table3_effect_sizes.csv"), index=False)
    print(f"  [OK] Table 3: Effect sizes with Hedges' g ({len(es_export)} models)")
    
    # Table 4: Meta-analysis summary
    summary = {
        "Analysis": ["Secondary Random-Effects Meta-Analysis", "Heterogeneity", "Meta-Regression", 
                      "Egger's Test", "Trim-and-Fill"],
        "Statistic": [
            f"d = {ma['pooled_effect']:.4f} [{ma['pooled_ci_lower']:.4f}, {ma['pooled_ci_upper']:.4f}]",
            f"I^2 = {ma['I_sq']:.1f}%, Q({ma['Q_df']}) = {ma['Q']:.2f}, tau^2 = {ma['tau_sq']:.4f}",
            f"beta1 = {mr['slope']:.4f}, R^2 = {mr['R_sq']:.4f}",
            f"intercept = {eg['intercept']:.4f}, t = {eg['intercept_t']:.4f}",
            f"Missing = {tf['n_missing']}, adj. d = {tf['adjusted_effect']:.4f}",
        ],
        "p-value": [
            f"{ma['pooled_p']:.4f}",
            f"{ma['Q_p']:.4f}",
            f"{mr['slope_p']:.4f}",
            f"{eg['intercept_p']:.4f}",
            "---",
        ],
    }
    pd.DataFrame(summary).to_csv(os.path.join(tables_dir, "table4_summary.csv"), index=False)
    print(f"  [OK] Table 4: Analysis summary")
    
    # Table 5: Correlation matrix
    corr.round(3).to_csv(os.path.join(tables_dir, "table5_correlations.csv"))
    print(f"  [OK] Table 5: Benchmark correlations")
    
    # Table 6: Leave-one-out results
    loo_df.round(4).to_csv(os.path.join(tables_dir, "table6_leave_one_out.csv"), index=False)
    print(f"  [OK] Table 6: Leave-one-out sensitivity")
    
    # Table 7: Bootstrap correlation CIs
    boot_df.round(4).to_csv(os.path.join(tables_dir, "table7_bootstrap_correlations.csv"), index=False)
    print(f"  [OK] Table 7: Bootstrap correlation CIs")
    
    # Table 8: Efficiency comparison
    # Table 8: Efficiency comparison

    # Table 9: Multilevel model results
    multilevel["moderator_table"].round(4).to_csv(
        os.path.join(tables_dir, "table9_multilevel_moderators.csv"), index=False)
    print(f"  + Table 9: Multilevel moderator table")
    
    # Table 10: Variance decomposition
    multilevel["variance_decomposition"].round(4).to_csv(
        os.path.join(tables_dir, "table10_variance_decomposition.csv"), index=False)
    print(f"  + Table 10: Variance decomposition")
    
    # Table 11: SE floor sensitivity
    multilevel["se_sensitivity"].round(4).to_csv(
        os.path.join(tables_dir, "table11_se_sensitivity.csv"), index=False)
    print(f"  + Table 11: SE floor sensitivity")
    pareto_data = robustness["pareto"]["data"][
        ["model", "d", "est_tflops", "eep_score", "pareto_optimal"]
    ].sort_values("eep_score", ascending=False).round(4)
    pareto_data.to_csv(os.path.join(tables_dir, "table8_efficiency.csv"), index=False)
    print(f"  [OK] Table 8: Efficiency Pareto analysis")

    # Table 12: Leave-one-benchmark-out multilevel sensitivity
    bloo_df.round(4).to_csv(
        os.path.join(tables_dir, "table12_benchmark_leave_one_out.csv"), index=False
    )
    print(f"  + Table 12: Benchmark leave-one-out multilevel sensitivity")

    # Table 13: Data provenance log
    # Table 13 reports the complete provenance record, not just the modelling
    # sample: every row of the verified table, including cells left blank where
    # no defensible source was found, and the models excluded from modelling
    # because their parameter counts are undisclosed. Each row carries the
    # document its score was taken from and a direct link to it.
    from data_collection import get_full_table

    provenance_cols = [
        "model", "benchmark", "score", "params_b", "vision_encoder",
        "llm_backbone", "training_strategy", "year", "source",
        "source_url", "source_venue", "peer_reviewed", "manuscript_ref",
        "score_source", "score_source_url", "score_source_type"
    ]
    full = get_full_table()
    provenance = full[provenance_cols].copy().sort_values(["model", "benchmark"])
    provenance["in_modelling_sample"] = [
        bool(((df["model"] == m) & (df["benchmark"] == b)).any())
        for m, b in zip(provenance["model"], provenance["benchmark"])
    ]
    provenance.to_csv(os.path.join(tables_dir, "table13_data_provenance.csv"), index=False)
    print(f"  + Table 13: Data provenance log ({len(provenance)} records, "
          f"{int(provenance['in_modelling_sample'].sum())} in the modelling sample)")
    
    # ================================================================
    # Step 12: Generate Figures (Original 7 + 4 Robustness)
    # ================================================================
    print_header("STEP 12: GENERATING FIGURES")
    generate_all_figures(results, figures_dir)
    generate_robustness_figures(results, robustness, figures_dir)

    # Save multivariate regression and open-weights Pareto for LaTeX extraction
    print("\n  [Extra] Saving multivariate regression results...")
    import json
    extra_results = {}
    extra_results["mr_multi"] = {k: v for k, v in mr_multi.items() if k != "wls_summary"}
    # Save bootstrap correlations for VQAv2 pairs
    boot_df = robustness["bootstrap_correlations"]
    vqav2_pairs = boot_df[(boot_df["benchmark_1"].str.contains("VQAv2")) | (boot_df["benchmark_2"].str.contains("VQAv2"))]
    extra_results["vqav2_bootstrap"] = vqav2_pairs.to_dict(orient="records")
    # Save open-weights Pareto models
    extra_results["pareto_open_models"] = pareto_open["pareto_models"]
    extra_results["pareto_open_n"] = pareto_open["n_pareto"]
    extra_results["pareto_open_best_eep"] = pareto_open["best_eep"]
    with open(os.path.join(tables_dir, "extra_results.json"), "w") as f:
        json.dump(extra_results, f, indent=2, default=str)
    print(f"  [OK] Extra results saved to {os.path.join(tables_dir, 'extra_results.json')}")
    
    # ================================================================
    # Final Summary
    # ================================================================
    print_header("ANALYSIS COMPLETE")
    print(f"""
  Key Findings:
  1. Overall pooled effect: d = {ma['pooled_effect']:.4f} (k = {ma['k']} models)
  2. Heterogeneity: I^2 = {ma['I_sq']:.1f}% ({hetero_level})
  3. Scale-performance correlation: r = {ov.get('r', 0):.3f} (p {format_p(ov.get('p', 1))})
  4. Meta-regression slope: beta = {mr['slope']:.4f} (p {format_p(mr['slope_p'])})
  5. Publication bias (Egger's): {bias}

  Robustness Checks:
    6. Hedges' g correlation with d: r = {corr_dg:.6f} (strong agreement)
  7. Leave-one-out max Deltad: {max_influence['delta_d']:+.4f} ({max_influence['excluded_model']})
  8. Open-source sensitivity: Deltad = {sens['delta_pooled_d']:+.4f}
  9. Pareto-optimal models: {pareto['n_pareto']}
  10. Influential models: {n_influential}
  
    Output Files:
    - {figures_dir}/ (12 figures)
    - {tables_dir}/ (13 CSV tables)
  
  [OK] Pipeline complete. All results are reproducible.
    """)
    
    return results, robustness


if __name__ == "__main__":
    results, robustness = main()

