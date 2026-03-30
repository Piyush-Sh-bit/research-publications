# Multi-Dimensional Benchmarking Framework (MDBF)

This repository contains the official dataset (embedded in the computing script) and the complete source code required to reproduce the numerical analysis and generate figures for our manuscript: *"A Multi-Dimensional Benchmarking Framework for Evaluating Multimodal Large Language Models Across Vision-Language Tasks"*.

## Repository Contents
To comply with the official Data Availability requirements, open-science principles, and ensure our proposed metrics are completely reproducible, we have publicly released the following files:

1. **`compute_results.py`** 
   - Contains the raw benchmark scores collected from our extensive meta-analysis of technical reports and public leaderboards.
   - Implements **Algorithm 1**, executing min-max normalization independent of specific benchmark scales.
   - Computes weighted aggregations to derive the Normalized Composite Scores across our five competency dimensions (Equations 1–3).
   - Calculates the Performance Gap Ratio (PGR) for proprietary versus open-source differences (Equation 4).

2. **`generate_figures.py`**
   - Implements the programmatic visualizations shown in the manuscript.
   - Generates the per-benchmark performance graphs, Competency Radar Charts (Figure 2), normalized score Heatmaps (Figure 3), and Statistical Error Distributions.

3. **`verify_values.py` and `verify_output.txt`**
   - Automated testing scripts and their corresponding output logs used to validate our computed MDBF metrics against normalized boundary constraints.

## How to use
To quickly reproduce the statistical output and regenerate all figures used in the manuscript, ensure you have Python installed along with `matplotlib`/`seaborn`/`pandas` (if required by the visualization script) and execute the following from the root directory:

```bash
python compute_results.py
python generate_figures.py
```
