# 📚 Scientific Research & Academic Publications

This monorepo is a central hub for datasets, evaluation methodologies, and supplemental algorithms supporting our research artifacts.

In accordance with open-science principles, data and algorithms are made publicly accessible to promote transparency and replicability.

## 📂 Projects

- **[MDBF_Framework](./MDBF_Framework)**
  - Scoring algorithms (Equations 1–4) and plotting scripts for the *Multi-Dimensional Benchmarking Framework for MLLMs* work.
  - See `MDBF_Framework/readme.md` for reproduction steps and details.

- **[Efficiency_vs._Efficacy](./Efficiency_vs._Efficacy)**
  - Materials for the **EEP Benchmark (Efficiency vs. Efficacy for Document Intelligence)**.
  - Includes:
    - `code.ipynb` (primary notebook for analysis/plots)
    - `eep_benchmark/` (core module: `dataset.py`, `engine.py`, `metrics.py`, `run_inference.py`)
    - `eep_benchmark/README_EXPERIMENT.md` (how to run real hardware experiments and export CSVs)
    - `fig_*.png` (generated analysis figures)

## 📝 Citation / Attribution

If you use this repository (code, figures, datasets, or derived results) in academic work, reports, or downstream projects, **please cite the associated paper(s) and link back to this repository**.

- If a project folder contains a `Citation` section and/or a `CITATION.cff`/BibTeX entry, use that.
- Otherwise, cite the relevant paper (when available) and include the repository URL in your references.

*(Project-specific citations will be added/updated as manuscripts are published or accepted.)
