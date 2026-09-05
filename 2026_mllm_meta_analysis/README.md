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

**Which part of the data was used.** For each model we extracted only the headline score reported by
that model's own authors for each of the seven benchmarks (MMBench, SEED-Bench, MM-Vet, MME, TextVQA,
POPE, VQAv2), as printed in the results tables of the source publication and under that publication's
own evaluation protocol. No benchmark images, test items, per-question responses or held-out splits
were accessed or used. Where a model did not report a benchmark, the value is recorded as missing and
never imputed — which is why 21 models yield 102, not 147, model–benchmark records.

**Where each value came from.** Every record in `Tables/table13_data_provenance.csv` carries its own
source webpage and publication status:

| Column | Meaning |
|--------|---------|
| `source` | Citation key of the publication the value was read from |
| `source_url` | Webpage of that publication |
| `source_venue` | Publication venue (or "not peer reviewed" where applicable) |
| `peer_reviewed` | Whether the source underwent peer review (`yes` / `no`) |
| `manuscript_ref` | Corresponding reference number in the manuscript |

These fields are generated from `SOURCE_REGISTRY` in `code/data_collection.py`, so they regenerate
with the rest of the pipeline rather than being maintained by hand.

**Peer-review status.** The 21 models trace to 17 distinct primary sources. **10 sources (12 models)**
are peer-reviewed publications (CVPR, NeurIPS, ICLR, ICML, ECCV, *Science China Information Sciences*).
The other **7 sources (9 models)** — GPT-4V, Gemini-Pro-V, Qwen-VL-Chat, Yi-VL-6B, Yi-VL-34B,
DeepSeek-VL-7B, Fuyu-8B, LLaVA-NeXT-7B and LLaVA-NeXT-13B — are self-reported technical reports,
system cards or project blog posts that have not undergone independent review. They are retained
because excluding them would remove the proprietary frontier models and several widely used
open-weight baselines, but they are flagged in the provenance table and in the paper's limitations.

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
