# Piyush Sharma — Academic Research Portfolio

Welcome to my central academic code repository. This space hosts the official code, datasets, and supplementary pipelines for my published and under-review academic journal papers.

Organizing my work this way ensures transparency, open science, and full reproducibility — in every project, each figure, table, and statistic regenerates from the bundled data with a single command.

---

## 📚 Publications & Pipelines

### 1. [Scale, Strategy, and Structure: A Multilevel Meta-Analysis of MLLMs](./2026_mllm_meta_analysis)
* **Status:** Under Review (2026)
* **Code Folder:** [`/2026_mllm_meta_analysis`](./2026_mllm_meta_analysis)
* **Description:** A two-level mixed-effects meta-analysis of **21 Multimodal Large Language Models (MLLMs)** across **7 visual-reasoning benchmarks**. Computes standardized effect sizes, decomposes performance variance (ICC), tests moderators (scale, training strategy, vision encoder), assesses publication bias (Egger's test, trim-and-fill), maps the accuracy–efficiency Pareto frontier, and runs a full robustness suite (leave-one-out, bootstrap intervals, Galbraith plot, influence diagnostics).
* **Tech Stack:** Python, Pandas, Statsmodels, SciPy, Matplotlib


---

## 🚀 How to Use This Repository

Each publication is self-contained within its own folder to prevent dependency conflicts. To run the code for a specific paper:

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/YourRepositoryName.git
   cd YourRepositoryName
   ```

2. Navigate to the specific publication's folder:
   ```bash
   cd 2026_mllm_meta_analysis          # or: cd 2026_mllm_228_single_harness
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Reproduce the analysis (each folder's own `README.md` lists the exact commands), for example:
   ```bash
   cd code
   python run_analysis.py              # 21-model study
   # python run_modern.py              # 228-model study
   ```

Every reported estimate, table, and figure regenerates from the bundled data with these commands.

---

## 👨‍💻 About Me

* **Author:** Piyush Sharma
* **Contact:** piyush.sh.rsh@gmail.com

Feel free to open an Issue with any questions about the methodology or codebase for any of the papers, or reach out to me directly via email.

---
*© 2026 Piyush Sharma. For code licensing, please refer to the specific `LICENSE` file provided within each individual project folder.*
