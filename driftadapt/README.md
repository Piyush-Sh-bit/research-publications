# DriftAdapt — gradient-free routed normalization memory for stable test-time adaptation

Reference code, results, and figures for the paper:

> **DriftAdapt: Gradient-Free Routed Normalization Memory for Stable Test-Time
> Adaptation under Non-Stationary and Recurrent Drift.**
> Piyush Sharma, School of Computer Science and Engineering, Lovely Professional
> University. *(under review)*

Test-time adaptation (TTA) updates a deployed model from unlabeled test data, but the
dominant gradient-based family (entropy minimization and its variants) is **fragile under
non-stationary drift**: confident-but-wrong predictions are self-reinforcing, and a single
continually-updated parameter set is overwritten when the distribution changes and must be
relearned when a previous state recurs. **DriftAdapt** is a **gradient-free** TTA method
that is stable by construction: it combines test-time feature normalization with a **CUSUM
drift detector** and a **bank of per-state normalization statistics** that are *reactivated,
not relearned,* when a state recurs. It performs no test-time optimization, so it has **no
learning rate to tune** and **cannot enter an error-reinforcement loop**, and it comes with a
**tracking-error (dynamic-regret) bound**. Everything runs **on CPU**.

## Key results (online accuracy %, official CIFAR-10-C severity 5, 5 seeds)

| Stream | Source | TTN | Tent | EATA | **DriftAdapt** |
|---|---|---|---|---|---|
| Sequential | 38.3 | 51.7 | 30.9 | 48.2 | **53.0** |
| Recurrent  | 38.2 | 51.7 | 34.7 | 49.3 | **52.8** |
| Gradual    | 64.2 | 74.8 | 62.6 | 76.1 | **76.1** |

- **DriftAdapt leads on all three streams** and is **learning-rate-invariant**; it matches
  the *best-tuned* Tent (52.8 vs 52.4 on recurrent) **with no tuning**.
- **Tent collapses** as its learning rate grows (52.4 → 46.1 → 34.7 → 26.3 → **19.4**);
  DriftAdapt/TTN have no learning rate.
- The CUSUM router is honestly **conservative on the noisy official corruptions**
  (recall 0.50–0.62), yet DriftAdapt stays best **because the per-state normalization memory,
  not the router, carries the adaptation**.
- Vs a **ReservoirTTA-style per-state model bank** (nearest prior art), DriftAdapt matches
  online accuracy and recurrence recovery while using **5.1× less per-state memory** and **no
  test-time backward pass** (~0.5 ms/step vs 0.9–1.3 ms).
- The ranking is **stable across severity**: repeating the grid at CIFAR-10-C severity 3 gives
  DriftAdapt 67.1 / 67.2 / 76.7 with Tent again collapsing.

## Repository layout

```
driftadapt/
├── src/
│   ├── driftadapt.py            # methods (source/ttn/tent/eata/DriftAdapt), streams, metrics,
│   │                            #   the gradient-free routed normalization memory + CUSUM router
│   ├── extract_features.py      # frozen ImageNet ResNet-18 features (official CIFAR-10-C)
│   ├── train_head.py            # trains the linear source head on clean features
│   ├── run_experiments.py       # 5-method × 3-stream × 5-seed grid, ablations, sweeps -> results.json
│   ├── extended_experiments.py  # batch-size / horizon robustness -> extended_results.json
│   ├── reservoir_experiments.py # ReservoirTTA-style model-bank comparison -> reservoir_results.json
│   └── make_figures.py          # regenerates the figures
├── results/                     # the actual run artifacts used in the paper
│   ├── results.json             # main grid + ablations + lr/drift sweeps
│   ├── extended_results.json    # batch-size and horizon robustness
│   └── reservoir_results.json   # nearest-prior-art (ReservoirTTA-style) comparison
├── figures/                     # trajectory, lr_robust, drift_freq, batch_sens, horizon (PNG)
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt          # torch (CPU ok), torchvision, numpy, matplotlib
```

Clean CIFAR-10 auto-downloads via torchvision. The **official CIFAR-10-C** corruption
benchmark must be downloaded once (Hendrycks & Dietterich, 2019):

```bash
cd src
curl -L -o CIFAR-10-C.tar https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
tar -xf CIFAR-10-C.tar        # -> ./CIFAR-10-C/{shot_noise,defocus_blur,fog,jpeg_compression,brightness}.npy
```

(Or point the `CIFAR10C_DIR` env var at an existing copy. `CIFAR10C_SEVERITY` defaults to 5.)

## Reproduce (CPU, ~minutes)

Run from `src/`, in order:

```bash
cd src
python extract_features.py     # frozen 512-d features -> cache.pt  (~2.5 min, one time)
python train_head.py           # linear source head -> head_std.pt
python run_experiments.py      # 5-seed grid + ablations + lr/drift sweeps -> results.json
python make_figures.py         # -> figures
python extended_experiments.py # batch-size / horizon robustness -> extended_results.json
python reservoir_experiments.py# ReservoirTTA-style comparison -> reservoir_results.json
```

`cache.pt` / `head_std.pt` / `data/` / `CIFAR-10-C/` are regenerated locally and git-ignored.
Every number in the paper regenerates from these scripts; results are seeded (5 seeds).

## Fair-comparison note

All adapting methods operate on the **same** frozen-backbone 512-d features and the same
affine surface — only the adaptation *algorithm* differs. DriftAdapt takes no test-time
gradient step (hence no learning rate); Tent/EATA use SGD at a shared rate. The main table
uses a shared lr = 0.1 for the gradient baselines; the lr-robustness figure reports the full
sweep and each baseline's best per-stream rate.

## Citation

```bibtex
@article{sharma2026driftadapt,
  title   = {DriftAdapt: Gradient-Free Routed Normalization Memory for Stable
             Test-Time Adaptation under Non-Stationary and Recurrent Drift},
  author  = {Sharma, Piyush},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the [MIT License](LICENSE).
