# Academic Research

Welcome to my central academic code repository. This space hosts the official code, datasets, and supplementary pipelines for my published and under-review academic journal papers.

Organizing my work this way ensures transparency, open science, and full reproducibility — in every project, each figure, table, and statistic regenerates from the bundled data with a single command.

---

## 📚 Publications & Pipelines

### 1. [Scale, Strategy, and Structure: A Multilevel Meta-Analysis of MLLMs](./2026_mllm_meta_analysis)
* **Status:** Under Review (2026)
* **Code Folder:** [`/2026_mllm_meta_analysis`](./2026_mllm_meta_analysis)
* **Description:** A two-level mixed-effects meta-analysis of **21 Multimodal Large Language Models (MLLMs)** across **7 visual-reasoning benchmarks**. Computes standardized effect sizes, decomposes performance variance (ICC), tests moderators (scale, training strategy, vision encoder), assesses publication bias (Egger's test, trim-and-fill), maps the accuracy–efficiency Pareto frontier, and runs a full robustness suite (leave-one-out, bootstrap intervals, Galbraith plot, influence diagnostics).
* **Tech Stack:** Python, Pandas, Statsmodels, SciPy, Matplotlib

### 2. [The Best Layer Is Predictable: An Unsupervised Information–Geometric Criterion for Representation-Layer Selection and Pruning in Language Models](./layeriq)
* **Status:** Under Review (2026)
* **Code Folder:** [`/layeriq`](./layeriq)
* **Description:** An unsupervised, label-free scalar — the **intrinsic dimension** of a frozen language model's per-layer representations, corroborated by representation-trajectory **curvature** — whose **argmax selects the downstream-optimal layer** with no labels and no training. Validated across **26 models (11 architectures, 160M–7B)**: beats the last-layer default in **72%** of cases (p = 7.5 × 10⁻⁵), recovering a **median 56%** of the oracle accuracy gap at zero labels; on the cells with real headroom (last-layer gap ≥ 2 pts) it recovers a **median 64%** (up to **+7 points**). Includes the diagnosed geometry-only **negative result** and a LayerIQ-guided **layer-pruning** application. Every headline number regenerates from the bundled results with `python analysis/headroom_analysis.py`.
* **Tech Stack:** Python, PyTorch, Transformers, scikit-learn, SciPy, NumPy

### 3. [DriftAdapt: Gradient-Free Routed Normalization Memory for Stable Test-Time Adaptation under Non-Stationary and Recurrent Drift](./driftadapt)
* **Status:** Under Review (2026)
* **Code Folder:** [`/driftadapt`](./driftadapt)
* **Description:** A **gradient-free** test-time adaptation (TTA) method that is **stable by construction**: it pairs test-time feature normalization with a **CUSUM drift detector** and a **bank of per-state normalization statistics** that are *reactivated, not relearned,* when a distribution recurs — so it has **no learning rate** to tune and cannot enter the entropy-collapse loop that sinks gradient-based TTA, and it comes with a **tracking-error (dynamic-regret) bound**. On the official **CIFAR-10-C** benchmark (5 seeds, CPU-only), DriftAdapt leads on sequential / recurrent / gradual streams (**53.0 / 52.8 / 76.1%**), matches best-tuned Tent with no tuning, and matches a ReservoirTTA-style per-state model bank while using **5.1× less per-state memory** and no test-time backward pass. Fully reproducible on CPU.
* **Tech Stack:** Python, PyTorch, torchvision, NumPy, Matplotlib

---

## 🚀 How to Use This Repository

Each publication is self-contained within its own folder to prevent dependency conflicts. To run the code for a specific paper:

1. Clone this repository:
   ```bash
   git clone https://github.com/Piyush-Sh-bit/research-publications.git
   cd research-publications
   ```

2. Navigate to the specific publication's folder:
   ```bash
   cd 2026_mllm_meta_analysis          # or: cd layeriq  /  cd driftadapt
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Reproduce the analysis (each folder's own `README.md` lists the exact commands), for example:
   ```bash
   # MLLM meta-analysis (21-model study):
   cd code
   python run_analysis.py

   # LayerIQ — regenerate every headline number from the bundled results (no GPU):
   python analysis/headroom_analysis.py
   # ...or re-run the full model sweep from scratch (Kaggle GPU):
   python run_all.py

   # DriftAdapt — reproduce on CPU (auto-downloads CIFAR-10; needs the CIFAR-10-C
   # benchmark — see driftadapt/README.md):
   cd src
   python extract_features.py && python train_head.py && python run_experiments.py && python make_figures.py
   ```

Every reported estimate, table, and figure regenerates from the bundled data with these commands.

---

## 👨‍💻 About Me
Author : Piyush Sharma
Feel free to open an Issue with any questions about the methodology or codebase for any of the papers, or reach out to me directly via email.

---
*© 2026 Piyush Sharma. For code licensing, please refer to the specific `LICENSE` file provided within each individual project folder.*
