"""Reproduce the paper's conditional-headroom result from the saved run artifacts.

The average accuracy gain of LayerIQ over the last-layer default (+1.2 points) is
misleading, because it is diluted by the many (model, task) cells where the last
layer is already optimal (nothing to gain). This script isolates the cells with a
real last-layer gap and shows what LayerIQ recovers *there*.

It reads results/results_{small,medium,high1,high2}.json (the same files the paper's
tables are built from) and prints:

  * the diluted all-cell mean/median accuracy gain (the misleading number), and
  * for each headroom threshold, on the subset of cells whose oracle beats the last
    layer by >= that margin: the mean accuracy gain, the median fraction of the gap
    recovered, and how often LayerIQ beats the last layer.

Usage:
    python analysis/headroom_analysis.py

Signal choice follows the paper: intrinsic dimension for encoder-family models,
representation-trajectory curvature for decoder-family models.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

# encoder-family name fragments (intrinsic dimension is the working signal for these;
# curvature is used for the decoder-family remainder)
ENC = ("bert-base", "roberta", "distilbert", "electra", "mpnet", "MiniLM",
       "e5-", "bge-", "gte-")


def family(model_name):
    leaf = model_name.split("/")[-1]
    return "enc" if any(k in leaf for k in ENC) else "dec"


def load_results():
    records = []
    for tag in ("small", "medium", "high1", "high2"):
        path = os.path.join(RESULTS, f"results_{tag}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                records += json.load(f)
    if not records:
        raise SystemExit(f"No results_*.json found in {RESULTS}. Run run_all.py first.")
    return records


def build_cells(records):
    """One cell per (model, classification task): LayerIQ's accuracy gain over the
    last layer, and the oracle headroom available."""
    cells = []
    for r in records:
        comp = r["components"]
        signal = np.asarray(
            comp["intrinsic_dimension"] if family(r["model"]) == "enc" else comp["curvature"],
            dtype=float)
        for task, v in r.get("classification", {}).items():
            if "per_layer" not in v:
                continue
            per_layer = np.asarray(v["per_layer"], dtype=float)
            n = min(len(signal), len(per_layer))
            pick = int(np.nanargmin(signal[:n]))   # LayerIQ = argmin of the signal
            last = per_layer[-1]
            oracle = per_layer[:n].max()
            gain = per_layer[pick] - last          # LayerIQ layer minus last layer
            gap = oracle - last                    # available headroom
            cells.append(dict(
                model=r["model"].split("/")[-1], family=family(r["model"]), task=task,
                gain=gain, gap=gap,
                recovered=(gain / gap if gap > 1e-9 else np.nan)))
    return cells


def main():
    cells = build_cells(load_results())
    gains = np.array([c["gain"] for c in cells])
    print(f"all cells: n={len(cells)}  "
          f"mean gain={gains.mean() * 100:+.2f} pts  median={np.median(gains) * 100:+.2f} pts")

    print("\nconditional on real last-layer headroom:")
    for thr in (0.02, 0.03, 0.05):
        sub = [c for c in cells if c["gap"] >= thr]
        if not sub:
            continue
        g = np.array([c["gain"] for c in sub])
        rec = np.array([c["recovered"] for c in sub])
        beats = np.mean([c["gain"] > 1e-9 for c in sub])
        print(f"  gap>={thr * 100:.0f}%: n={len(sub):3d}  "
              f"mean gain={g.mean() * 100:+.2f} pts  "
              f"median recovered={np.nanmedian(rec) * 100:.0f}%  "
              f"beats-last={beats * 100:.0f}%")

    print("\ntop-10 single cells by accuracy gain:")
    for c in sorted(cells, key=lambda x: -x["gain"])[:10]:
        print(f"  {c['model']:26s} {c['task']:5s} "
              f"gain={c['gain'] * 100:+.1f} pts  gap={c['gap'] * 100:.1f}  "
              f"recovered={c['recovered'] * 100:.0f}%")


if __name__ == "__main__":
    main()
