# LayerIQ — the best representation layer of a frozen LM is predictable, with no labels

Reference code and results for the paper:

> **The Best Layer Is Predictable: An Unsupervised Information–Geometric Criterion
> for Representation-Layer Selection and Pruning in Language Models.**
> Piyush Sharma, School of Computer Science and Engineering, Lovely Professional
> University. *(under review)*

The last hidden layer of a pretrained language model is rarely its best
general-purpose representation — intermediate layers often transfer better — but
existing evidence is *descriptive*: it says a good layer exists, not **which one**
to use without first running a labelled downstream evaluation. **LayerIQ** is a
single, parameter-free, label-free scalar computed per layer from at most a few
hundred unlabelled sentences whose **argmax selects the best layer**.

## The criterion

```
LayerIQ(ℓ) = intrinsic dimension of layer ℓ's representations   (minimiser = best layer)
```

The minimiser selects the most compressed, semantically concentrated layer. A
second, architecture-complementary signal — **representation-trajectory
curvature** — corroborates it. In practice: **intrinsic dimension for encoders,
curvature for decoders**. See [`src/metrics.py`](src/metrics.py) for both.

The paper also reports an instructive **negative result**: the natural
geometry-only version (a geometric mean of effective rank, isotropy, and a
self-retrieval stability term) *fails* — its argmax collapses onto the input
layer and anti-correlates with probe quality — because those signals carry a
monotone-depth confound. That failing criterion is preserved in
[`results/results_failure_geometry_v1.csv`](results/results_failure_geometry_v1.csv).

## Key results (26 models, 11 architectures, 160M–7B)

- Selects the best probing-classification layer with **rank agreement +0.69
  (encoders) / +0.73 (decoders)**, up to **+0.94** on Qwen2.5-7B.
- **Beats the last-layer default in 72% of cases** (p = 7.5 × 10⁻⁵).
- At zero labels, recovers a **median 56%** of the accuracy gap an exhaustive
  labelled layer search would find.
- **Conditional headroom:** on the 44 / 78 (model, task) cells with a non-trivial
  last-layer gap (≥ 2 points), LayerIQ adds **+1.8 points on average** and recovers
  a **median 64%** of the available gap (up to **+7 points** on gte-base / TREC);
  it is correctly inert where the last layer is already best.
- **Scope boundary (honest):** on embedding-tuned models the two *similarity*
  tasks (STS, paraphrase retrieval) peak *exactly* at the final layer
  (last-layer gap 0.000 on 8/8), so no selection is needed there. LayerIQ
  predicts **linear-probe separability, not similarity geometry**.

Reproduce the conditional-headroom numbers from the raw JSON with
[`analysis/headroom_analysis.py`](analysis/headroom_analysis.py).

## Repository layout

```
layeriq/
├── src/
│   ├── config.py         # model tiers + all hyperparameters
│   ├── metrics.py        # LayerIQ + corroborating signals  (core contribution)
│   ├── extraction.py     # frozen-model hidden-state extraction (inference only)
│   ├── data.py           # WikiText (unlabelled), STS-B, SST-2, TREC, SUBJ, MRPC
│   ├── evaluation.py     # downstream validation + rank agreement / regret
│   ├── pruning.py        # LayerIQ-guided layer pruning (application)
│   └── run_experiment.py # end-to-end orchestrator
├── run_all.py            # ONE script that reproduces every tier -> results/
├── results/              # the actual run artifacts used in the paper
│   ├── results_{small,medium,high1,high2}.json    # full per-layer panels
│   ├── results_{...}_summary.csv                  # one row per (model, task)
│   ├── results_failure_geometry_v1.csv            # the diagnosed negative
│   └── log.txt                                    # full run log
├── analysis/
│   └── headroom_analysis.py   # reproduces the conditional-headroom result
├── requirements.txt
└── LICENSE
```

## Reproduce

### On Kaggle (as run for the paper; GPU T4 / P100)

1. Upload the contents of [`src/`](src/) **plus** `requirements.txt` as a Kaggle
   **Dataset** (any name).
2. New Notebook → **Accelerator = GPU**, **Internet = ON** → **Add Input** to
   attach that dataset.
3. Paste [`run_all.py`](run_all.py) into one cell → **Run All**. It auto-discovers
   the code regardless of the dataset name, runs every tier in order, and writes
   `results_*.json` + `results_*_summary.csv` to `/kaggle/working/results/`.

The full sweep is ~4 h and checkpoints after every model, so an interrupted run
still leaves every finished model on disk. Comment out tiers in the `RUNS` list at
the bottom of `run_all.py` to run a subset (keep only `small` for a ~45-min
reproduction). The `high` tier (7B, 4-bit) is the slow one. All models are
ungated (no Hugging Face token); any that fails to load is skipped automatically.

### Local smoke test (CPU, tiny)

```bash
pip install -r requirements.txt
cd src
python -c "from config import ExperimentConfig; import run_experiment as R; \
R.main(ExperimentConfig(models=['openai-community/gpt2'], n_unlabeled=64))"
```

### Reproduce the paper's headline analysis from the saved results

```bash
python analysis/headroom_analysis.py     # reads results/*.json, prints the headroom table
```

## Results file format

`results_<tier>_summary.csv` — one row per (model, task) with `oracle_layer`,
`last_layer_gap`, and the selected layer + regret + rank agreement for both
**intrinsic dimension** and **curvature**. Read it as: LayerIQ works when
`*_rank_agree → +1` and `*_regret` is small. The `.json` files hold the full
per-layer panel behind each summary row.

## Citation

```bibtex
@article{sharma2026layeriq,
  title   = {The Best Layer Is Predictable: An Unsupervised Information--Geometric
             Criterion for Representation-Layer Selection and Pruning in Language Models},
  author  = {Sharma, Piyush},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the [MIT License](LICENSE).
