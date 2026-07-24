"""Downstream evaluation used ONLY to validate LayerIQ's predictions.

For each layer we measure true downstream quality:
  - STS-B: Spearman correlation between cosine(emb1, emb2) and gold score.
  - Classification: accuracy of a logistic-regression probe on frozen embeddings.

We then check whether the layer LayerIQ *predicted* (without labels) matches /
approaches the empirically best layer, and report rank correlation across all
layers between LayerIQ and downstream quality.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def sts_score_per_layer(
    emb1_layers: List[torch.Tensor],
    emb2_layers: List[torch.Tensor],
    gold: List[float],
) -> List[float]:
    scores = []
    gold = np.asarray(gold)
    for z1, z2 in zip(emb1_layers, emb2_layers):
        a = torch.nn.functional.normalize(z1.float(), dim=1)
        b = torch.nn.functional.normalize(z2.float(), dim=1)
        cos = (a * b).sum(1).numpy()
        rho = spearmanr(cos, gold).correlation
        scores.append(float(rho))
    return scores


def probe_accuracy_per_layer(
    train_layers: List[torch.Tensor],
    y_train: List[int],
    test_layers: List[torch.Tensor],
    y_test: List[int],
    seed: int = 0,
) -> List[float]:
    accs = []
    for ztr, zte in zip(train_layers, test_layers):
        sc = StandardScaler()
        Xtr = sc.fit_transform(ztr.numpy())
        Xte = sc.transform(zte.numpy())
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(Xtr, y_train)
        accs.append(float(clf.score(Xte, y_test)))
    return accs


def rank_agreement(layeriq: np.ndarray, downstream: List[float]):
    """Spearman between LayerIQ and true downstream score across layers."""
    rho = spearmanr(np.asarray(layeriq), np.asarray(downstream)).correlation
    return float(rho)


def selection_regret(layeriq: np.ndarray, downstream: List[float]):
    """Gap between best-possible downstream score and the score at the layer
    LayerIQ selected. 0 means LayerIQ picked the optimal layer."""
    d = np.asarray(downstream)
    picked = int(np.argmax(layeriq))
    return float(d.max() - d[picked]), picked, int(d.argmax())


def last_layer_gap(downstream: List[float]):
    """How much the default last-layer choice loses vs. the best layer."""
    d = np.asarray(downstream)
    return float(d.max() - d[-1])


def retrieval_metrics_per_layer(
    query_layers: List[torch.Tensor],
    pool_layers: List[torch.Tensor],
):
    """Paraphrase retrieval per layer. query_layers[l], pool_layers[l] are (n,d);
    query i's gold match is pool i. Returns (accuracy@1 list, MRR list) per layer."""
    accs, mrrs = [], []
    for zq, zp in zip(query_layers, pool_layers):
        a = torch.nn.functional.normalize(zq.float(), dim=1)
        b = torch.nn.functional.normalize(zp.float(), dim=1)
        sim = a @ b.t()                       # (n, n) cosine similarity
        gold = sim.diag().unsqueeze(1)        # score of the correct match
        ranks = (sim > gold).sum(dim=1) + 1   # rank of the gold match (1 = top)
        accs.append(float((ranks == 1).float().mean()))
        mrrs.append(float((1.0 / ranks.float()).mean()))
    return accs, mrrs
