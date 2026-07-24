"""LayerIQ: an unsupervised, label-free criterion for ranking the
representation layers of a language model.

LayerIQ(l) is a single scalar per layer l, computed from a small unlabeled
corpus. It aggregates three complementary, parameter-free signals:

    1. Effective rank (RankMe)         -> dimensional utilisation / diversity.
    2. Isotropy (1 - mean anisotropy)  -> embeddings spread well for cosine.
    3. Perturbation self-retrieval     -> semantic robustness (our new signal).

Each signal is min-max normalised across layers, then combined with a
parameter-free geometric mean. The argmax over layers is the predicted
best layer for embedding extraction.

All functions operate on a pooled representation matrix Z of shape
(n_sentences, hidden_dim) for a single layer.
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Signal 1: Effective rank (RankMe)
# --------------------------------------------------------------------------- #
def effective_rank(Z: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe effective rank: exp of the entropy of the normalised singular
    value spectrum. High value => representation uses many dimensions.

    Garrido et al., 2023 ("RankMe"). Computed in float32 on CPU for stability.
    """
    Zc = Z.float() - Z.float().mean(0, keepdim=True)
    # singular values of the centred matrix
    s = torch.linalg.svdvals(Zc)
    p = s / (s.sum() + eps)
    p = p[p > 0]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())


# --------------------------------------------------------------------------- #
# Signal 2: Isotropy (inverse anisotropy)
# --------------------------------------------------------------------------- #
def isotropy(Z: torch.Tensor, n_pairs: int = 20000, seed: int = 0) -> float:
    """1 - mean absolute cosine similarity over random sentence pairs.

    Anisotropic spaces (all vectors point the same way) are poor for cosine
    similarity. Value in [0, 1]; higher is better.
    """
    g = torch.Generator().manual_seed(seed)
    Zn = torch.nn.functional.normalize(Z.float(), dim=1)
    n = Zn.shape[0]
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    mask = i != j
    cos = (Zn[i[mask]] * Zn[j[mask]]).sum(1)
    return float((1.0 - cos.abs().mean()).item())


# --------------------------------------------------------------------------- #
# Signal 3: Perturbation self-retrieval stability  (NEW, our contribution)
# --------------------------------------------------------------------------- #
def self_retrieval_stability(
    Z_clean: torch.Tensor, Z_perturbed: torch.Tensor, k: int = 1
) -> float:
    """Self-retrieval accuracy under input perturbation.

    Z_clean[i] and Z_perturbed[i] are representations of the same sentence,
    where the perturbed version had a fraction of its tokens dropped/masked.
    We query each clean embedding against the bank of perturbed embeddings and
    measure how often the matching index is within the top-k.

    A layer that encodes stable *semantic* content (rather than surface form)
    keeps a sentence close to its own perturbed twin even when other sentences
    crowd the space -> high stability. This needs NO labels.
    """
    Zc = torch.nn.functional.normalize(Z_clean.float(), dim=1)
    Zp = torch.nn.functional.normalize(Z_perturbed.float(), dim=1)
    sim = Zc @ Zp.t()                       # (n, n) cosine similarity
    n = sim.shape[0]
    topk = sim.topk(k, dim=1).indices       # (n, k)
    target = torch.arange(n).unsqueeze(1)
    hit = (topk == target).any(dim=1).float()
    return float(hit.mean().item())


# --------------------------------------------------------------------------- #
# Signal 4 (v2): Matrix-based (von Neumann) entropy
# --------------------------------------------------------------------------- #
def matrix_entropy(Z: torch.Tensor, alpha: float = 1.0, eps: float = 1e-8) -> float:
    """Matrix-based (von Neumann / Renyi-alpha) entropy of the representation.

    Following Skean et al. (2025), we L2-normalise the sentence embeddings, form
    the K x K Gram matrix G = Z Z^T / K (a unit-trace density matrix), and take
    the entropy of its eigenvalue spectrum. Unlike RankMe -- which uses the
    feature covariance and, empirically, trends monotonically with depth -- this
    sample-Gram entropy is documented to be non-monotone, peaking where the
    representation is most informative. alpha=1 gives the von Neumann entropy.
    Label-free; O(K^3) for K sentences (cheap for K <= a few hundred).
    """
    Zn = torch.nn.functional.normalize(Z.float(), dim=1)
    n = Zn.shape[0]
    G = (Zn @ Zn.t()) / n                      # trace(G) = 1
    ev = torch.linalg.eigvalsh(G).clamp(min=0)
    ev = ev / (ev.sum() + eps)
    ev = ev[ev > eps]
    if abs(alpha - 1.0) < 1e-6:
        H = -(ev * torch.log(ev)).sum()
    else:
        H = (1.0 / (1.0 - alpha)) * torch.log((ev ** alpha).sum() + eps)
    return float(H.item())


# --------------------------------------------------------------------------- #
# Signal 5 (v2): Intrinsic dimension (TwoNN)
# --------------------------------------------------------------------------- #
def intrinsic_dimension_twonn(Z: torch.Tensor, discard_frac: float = 0.1,
                              eps: float = 1e-12) -> float:
    """Intrinsic dimension via the TwoNN estimator (Facco et al., 2017).

    For each point, mu = r2 / r1 is the ratio of distances to its 2nd and 1st
    nearest neighbours. Under a locally-uniform density, log(mu) is exponential
    with rate equal to the intrinsic dimension d, estimated by a through-origin
    fit of -log(1 - F(mu)) against log(mu). Label-free; documented to peak in
    interior layers where semantic structure concentrates (Valeriani et al.,
    2023) -- a non-monotone signal the v1 panel lacked.
    """
    X = Z.float()
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    vals, _ = torch.topk(d, k=2, dim=1, largest=False)   # r1, r2 per point
    mu = (vals[:, 1] / (vals[:, 0] + eps)).cpu().numpy()
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1.0 + 1e-6)])
    if len(mu) < 10:
        return float("nan")
    keep = max(int(len(mu) * (1.0 - discard_frac)), 10)
    mu = mu[:keep]
    n = len(mu)
    x = np.log(mu)
    y = -np.log(1.0 - np.arange(1, n + 1) / (n + 1))       # empirical CDF
    return float((x * y).sum() / ((x * x).sum() + eps))     # slope through origin


# --------------------------------------------------------------------------- #
# Aggregation into the LayerIQ score
# --------------------------------------------------------------------------- #
def _minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + eps)


def layeriq_scores(
    eff_rank: list[float],
    iso: list[float],
    stab: list[float],
) -> np.ndarray:
    """Combine the three per-layer signal vectors into LayerIQ.

    Each signal is min-max normalised across layers (so all live in [0, 1]),
    then aggregated with a parameter-free geometric mean. The geometric mean
    penalises a layer that is weak on any single axis -- a good embedding layer
    must be diverse AND isotropic AND robust.
    """
    er = _minmax(eff_rank)
    it = _minmax(iso)
    st = _minmax(stab)
    eps = 1e-6
    geo = np.cbrt((er + eps) * (it + eps) * (st + eps))
    return geo


def predicted_best_layer(scores: np.ndarray) -> int:
    return int(np.argmax(scores))
