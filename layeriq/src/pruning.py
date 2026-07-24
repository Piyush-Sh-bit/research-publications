"""Application 2: LayerIQ-guided layer pruning.

Premise: layers with low LayerIQ contribute little usable representation
quality. We greedily keep the top-k layers by LayerIQ and measure how much
downstream quality is retained vs. keeping the last k layers (the naive
"drop early layers" baseline) and vs. random subsets.

This is fully label-free at selection time; labels are only used to *report*
retained quality.
"""
from __future__ import annotations

import numpy as np


def pruning_curve(layeriq, downstream, baseline="last"):
    """For each budget k=1..L, retained downstream quality when we keep the
    k layers chosen by a strategy.

    Returns dict of strategy -> list over k of best retained score among kept.
    (We report the max downstream score among kept layers, i.e. the quality of
    the best surviving embedding layer under each budget.)
    """
    layeriq = np.asarray(layeriq)
    downstream = np.asarray(downstream)
    L = len(layeriq)
    order_iq = np.argsort(-layeriq)            # high LayerIQ first
    order_last = np.arange(L)[::-1]            # last layers first
    rng = np.random.default_rng(0)

    out = {"layeriq": [], "last": [], "random": []}
    for k in range(1, L + 1):
        keep_iq = order_iq[:k]
        keep_last = order_last[:k]
        rand_scores = []
        for _ in range(20):
            keep_r = rng.choice(L, size=k, replace=False)
            rand_scores.append(downstream[keep_r].max())
        out["layeriq"].append(float(downstream[keep_iq].max()))
        out["last"].append(float(downstream[keep_last].max()))
        out["random"].append(float(np.mean(rand_scores)))
    return out


def area_under_retention(curve):
    """Higher AUC => better quality retention across all pruning budgets."""
    return {k: float(np.trapz(v) / len(v)) for k, v in curve.items()}
