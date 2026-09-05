# MLLM Visual Reasoning Meta-Analysis

**Title:** *Scale, Strategy, and Structure: A Multilevel Meta-Analysis of Multimodal Large Language Model Performance on Visual Reasoning Benchmarks*
**Author:** Piyush Sharma

## Overview

This repository contains the complete code and data for a multilevel meta-analysis of Multimodal Large Language Model (MLLM) performance across visual reasoning benchmarks.

## Project Structure

```text
├── code/
│   ├── data_collection.py          # Compiled benchmark data from 21 MLLM papers
│   ├── extract_results.py          # Extracts numerical results for reporting
│   ├── multilevel_analysis.py      # Meta-analysis multilevel statistical methods
│   ├── robustness_analysis.py      # Additional robustness checks and sensitivity
│   ├── run_analysis.py             # Main pipeline entry point
│   ├── statistical_analysis.py     # Random-effects meta-analysis & meta-regression
│   ├── visualization.py            # Primary publication-quality figure generation
│   └── visualization_robustness.py # Generates figures for robustness checks
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

*(Note: Running the analysis script will automatically generate a `results/` directory containing all output figures and results CSV tables.)*

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
python code/run_analysis.py
```

This will:
- Load benchmark data from 21 MLLMs across 7 benchmarks
- Normalize scores and compute standardized effect sizes
- Fit a multilevel mixed-effects model with benchmark scores nested within models
- Run DerSimonian-Laird random-effects meta-analysis as a secondary sensitivity check
- Perform subgroup analyses (training strategy, encoder family, model scale)
- Conduct meta-regression (model scale → performance)
- Test for publication bias (Egger's test, trim-and-fill)
- Compute inter-benchmark Spearman correlations
- Run extensive robustness analyses (Hedges' g comparison, leave-one-out sensitivity, excluding proprietary models, efficiency Pareto frontier)
- Output influence diagnostics and outlier detection (Galbraith plot)
- Generate 12 publication-quality figures
- Export 13 results CSV tables containing full multilevel, robustness, and data provenance logs

## Data Availability and Provenance

All data analysed here are **third-party (secondary) data**: benchmark scores previously published
by the developers of each model. No new model evaluations were run for this study, and **no benchmark
dataset is redistributed in this repository**.

**Which part of the data was used.** For each model–benchmark pair we recorded only the single
publicly reported headline score for the seven benchmarks (MMBench, SEED-Bench, MM-Vet, MME, TextVQA,
POPE, VQAv2) — taken from the model's own publication where that publication reported the benchmark,
and otherwise from the benchmark's own evaluation tables or a public leaderboard snapshot, since
several benchmarks postdate the release of the earlier models. No benchmark images, test items,
per-question responses or held-out splits were accessed or used. Where a model had no publicly
reported score for a benchmark, the value is recorded as missing and never imputed — which is why
21 models yield 102, not 147, model–benchmark records.

*Where the **model** is described:*

| Column | Meaning |
|--------|---------|
| `source` | Citation key of the model's publication |
| `source_url` | Webpage of that publication |
| `source_venue` | Publication venue (or "not peer reviewed" where applicable) |
| `peer_reviewed` | Whether that publication underwent peer review (`yes` / `no`) |
| `manuscript_ref` | Corresponding reference number in the manuscript |

*Where the **score** was taken from:*

| Column | Meaning |
|--------|---------|
| `score_source` | Citation key of the document this value was taken from |
| `score_source_url` | Direct link to that document |
| `score_source_type` | `model_paper`, `benchmark_paper`, or `leaderboard` |

Every one of the 102 records carries a source and a working link. The split is 76 from the
model's own publication, 19 from a public leaderboard, and 7 from a benchmark paper. A score is
attributed to the model's own publication only where that publication could have reported it;
where the benchmark postdates the model paper, or the release document carries no benchmark
tables (a system card or blog post), the source is the benchmark's own paper or the relevant
leaderboard instead.

Sources used: the 17 model publications listed in `SOURCE_REGISTRY`, the MM-Vet and SEED-Bench
papers, and three public leaderboards — MME
(`github.com/BradyFU/Awesome-Multimodal-Large-Language-Models`), MMBench
(`mmbench.opencompass.org.cn/leaderboard`) and OpenCompass OpenVLM
(`huggingface.co/spaces/opencompass/open_vlm_leaderboard`).

All of these fields are generated from `SOURCE_REGISTRY` and `SCORE_PROVENANCE` in
`code/data_collection.py`, so they regenerate with the pipeline rather than being hand-maintained.

> **Note on one corrected value.** GPT-4V's MM-Vet score was previously recorded as 56.8. That
> figure is MM-ReAct-GPT-4's spatial-awareness sub-score in the MM-Vet paper, not GPT-4V's total;
> it has been corrected to 67.7, the total reported for GPT-4V in that paper. With the corrected
> value the leave-one-benchmark-out multilevel fit does not converge when MMBench, MME or TextVQA
> is excluded (rank-deficient design given the two-model RLHF subgroup); those rows are reported
> as non-converged in Table 12 rather than omitted.

## Statistical Methods

| Method | Implementation | Purpose |
|--------|---------------|---------|
| DerSimonian-Laird | `statistical_analysis.py` | Random-effects pooling |
| Multilevel mixed-effects | `multilevel_analysis.py` | Benchmark-within-model synthesis |
| Cochran's Q, I², τ² | `statistical_analysis.py` | Heterogeneity assessment |
| Meta-regression (WLS) | `statistical_analysis.py` | Scale as moderator |
| Egger's test & Trim-and-fill | `statistical_analysis.py` | Publication bias assessment |
| Spearman ρ | `statistical_analysis.py` | Benchmark correlations |
| Leave-one-out / Sensitivity | `robustness_analysis.py` | Sensitivity analysis and robustness checks |
| Influence diagnostics | `robustness_analysis.py` | Cook's distance, DFBETAS, Galbraith plots |
| Efficiency Pareto frontier | `robustness_analysis.py` | Computing Pareto optimal models and EEP scores |

## Key Results
- **21 models** analyzed across **7 benchmarks** (102 observations)
- **Multilevel ICC = 0.8005**: Strong clustering at the model level in the primary analysis
- **Training strategy** appears to be an important moderator, but the RLHF subgroup is small
- **Model scale** remains significant in the primary mixed-effects model (β≈0.73, p≈0.016)
- **DL aggregate analysis** still shows I² = 96.3% and near-zero pooled effect as a secondary check
- **Benchmarks** show strong concordance (ρ = 0.63–0.95)

## Dependencies

- Python >= 3.9
- NumPy, Pandas, SciPy, Matplotlib, Seaborn, Statsmodels, Scikit-learn
