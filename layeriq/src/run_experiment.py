"""End-to-end LayerIQ experiment.

Pipeline per model:
  1. Compute LayerIQ(l) from an UNLABELED corpus  (no labels, no training).
  2. Compute true downstream quality per layer (STS-B Spearman, probe accuracy).
  3. Report:
       - rank agreement (Spearman) between LayerIQ and each downstream metric,
       - selection regret of LayerIQ vs the oracle best layer,
       - improvement of LayerIQ-selected layer over the default last layer,
       - LayerIQ-guided pruning curve.

Outputs JSON + CSV into config.out_dir.
"""
from __future__ import annotations

import csv
import gc
import json
import os

import numpy as np
import torch

from config import DEFAULT, ExperimentConfig
import data
import extraction as ex
import metrics as M
import evaluation as ev


def compute_layeriq(sentences, tok, model, cfg: ExperimentConfig):
    """Return (layeriq_vector, components_dict)."""
    clean = ex.extract_all_layers(
        sentences, tok, model,
        max_length=cfg.max_length, batch_size=cfg.batch_size,
        perturb=False,
    )
    perturbed = ex.extract_all_layers(
        sentences, tok, model,
        max_length=cfg.max_length, batch_size=cfg.batch_size,
        perturb=True, token_dropout_p=cfg.token_dropout, seed=cfg.perturb_seed,
    )
    eff = [M.effective_rank(z) for z in clean]
    iso = [M.isotropy(z, seed=cfg.seed) for z in clean]
    stab = [M.self_retrieval_stability(zc, zp, k=cfg.knn_k)
            for zc, zp in zip(clean, perturbed)]
    # v2 candidate signals: non-monotone, better-grounded; recorded per layer so
    # the criterion can be (re)designed offline from the dump, no extra GPU runs.
    ment = [M.matrix_entropy(z, alpha=1.0) for z in clean]
    idim = [M.intrinsic_dimension_twonn(z) for z in clean]
    # v3 candidate: token-trajectory curvature (Skean et al.) -- a new signal to
    # test whether it beats intrinsic dimension as the selector.
    curv = ex.curvature_per_layer(sentences, tok, model,
                                  max_length=cfg.max_length, batch_size=cfg.batch_size)
    scores = M.layeriq_scores(eff, iso, stab)   # v1 (kept, to showcase the failure)
    return scores, {"effective_rank": eff, "isotropy": iso, "stability": stab,
                    "matrix_entropy": ment, "intrinsic_dimension": idim,
                    "curvature": curv}


def run_model(name: str, cfg: ExperimentConfig) -> dict:
    print(f"\n=== {name} ===")
    tok, model = ex.load_model(name, load_in_4bit=cfg.load_in_4bit, device=cfg.device)

    # --- (1) unsupervised criterion ---
    unlabeled = data.load_unlabeled(
        cfg.unlabeled_corpus, cfg.unlabeled_subset, cfg.n_unlabeled, cfg.seed
    )
    layeriq, components = compute_layeriq(unlabeled, tok, model, cfg)
    n_layers = len(layeriq)

    result = {"model": name, "n_layers": n_layers,
              "layeriq": layeriq.tolist(), "components": components,
              "predicted_layer": M.predicted_best_layer(layeriq)}

    # --- (2/3) downstream validation: STS-B ---
    # Wrapped so a single dataset/hub failure never discards the LayerIQ result
    # (the core contribution) or the other tasks that did succeed.
    try:
        s1, s2, gold = data.load_sts()
        s1, s2, gold = s1[:cfg.sts_limit], s2[:cfg.sts_limit], gold[:cfg.sts_limit]
        z1 = ex.extract_all_layers(s1, tok, model, cfg.max_length, cfg.batch_size)
        z2 = ex.extract_all_layers(s2, tok, model, cfg.max_length, cfg.batch_size)
        sts = ev.sts_score_per_layer(z1, z2, gold)
        result["sts"] = {
            "per_layer": sts,
            "rank_agreement": ev.rank_agreement(layeriq, sts),
            "regret": ev.selection_regret(layeriq, sts)[0],
            "picked_vs_oracle": ev.selection_regret(layeriq, sts)[1:],
            "last_layer_gap": ev.last_layer_gap(sts),
        }
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] STS-B failed: {e}")
        result["sts"] = {"error": str(e)}

    # --- downstream validation: classification probes ---
    result["classification"] = {}
    for task in cfg.cls_tasks:
        try:
            xtr, ytr, xte, yte = data.load_classification(task, cfg.cls_n_train, cfg.cls_n_test)
            ztr = ex.extract_all_layers(xtr, tok, model, cfg.max_length, cfg.batch_size)
            zte = ex.extract_all_layers(xte, tok, model, cfg.max_length, cfg.batch_size)
            acc = ev.probe_accuracy_per_layer(ztr, ytr, zte, yte, cfg.seed)
            result["classification"][task] = {
                "per_layer": acc,
                "rank_agreement": ev.rank_agreement(layeriq, acc),
                "regret": ev.selection_regret(layeriq, acc)[0],
                "picked_vs_oracle": ev.selection_regret(layeriq, acc)[1:],
                "last_layer_gap": ev.last_layer_gap(acc),
            }
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] classification task '{task}' failed: {e}")
            result["classification"][task] = {"error": str(e)}

    # --- downstream APPLICATION: paraphrase retrieval (semantic search) ---
    try:
        q, pool = data.load_retrieval(cfg.retrieval_task, cfg.retrieval_limit)
        zq = ex.extract_all_layers(q, tok, model, cfg.max_length, cfg.batch_size)
        zp = ex.extract_all_layers(pool, tok, model, cfg.max_length, cfg.batch_size)
        acc, mrr = ev.retrieval_metrics_per_layer(zq, zp)
        result["retrieval"] = {
            "task": cfg.retrieval_task, "metric": "acc@1",
            "per_layer": acc, "per_layer_mrr": mrr,
            "rank_agreement": ev.rank_agreement(layeriq, acc),
            "last_layer_gap": ev.last_layer_gap(acc),
        }
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] retrieval failed: {e}")
        result["retrieval"] = {"error": str(e)}

    del model, tok      # must drop the refs HERE, in the owning frame
    _free()
    return result


def _free():
    """Reclaim GPU memory before the next model loads.

    Dropping the reference is not enough on its own: HF modules hold reference
    cycles, so the model survives plain refcounting until a gc pass runs, and
    empty_cache() would then free nothing. Callers must `del` their own refs
    first -- doing it inside this function would only unbind a local here.
    Without this, memory accumulates and a later (often smaller) model OOMs.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _purge_hf_cache(repo_id: str) -> None:
    """Evict a model's downloaded weights from the HF cache.

    GPU memory is not the only thing that accumulates: every model stays on DISK
    after use, and the full sweep pulls ~90GB of weights (the 7B tier alone is
    ~56GB), which overruns Kaggle's disk and kills the session with "high /tmp
    usage / out of system disk". Each model is read exactly once, so its weights
    are dead the moment its result is recorded.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        hashes = [rev.commit_hash for repo in cache.repos
                  if repo.repo_id == repo_id for rev in repo.revisions]
        if hashes:
            strategy = cache.delete_revisions(*hashes)
            freed = strategy.expected_freed_size_str
            strategy.execute()
            print(f"  [cache] evicted {repo_id} ({freed})")
    except Exception as e:  # noqa: BLE001 -- never let cleanup kill the sweep
        print(f"  [cache] could not purge {repo_id}: {e}")


def main(cfg: ExperimentConfig = DEFAULT):
    os.makedirs(cfg.out_dir, exist_ok=True)
    all_results = []
    ckpt = os.path.join(cfg.out_dir, "results.json")
    for i, name in enumerate(cfg.models, 1):
        try:
            all_results.append(run_model(name, cfg))
        except Exception as e:  # keep going across models
            print(f"[WARN] {name} failed: {e}")
        # Reclaim here rather than inside the except: until that block exits the
        # interpreter still holds the exception, whose traceback pins run_model's
        # frame and the failed model with it, so collecting in there frees
        # nothing and one OOM cascades into every model that follows.
        _free()
        if cfg.purge_hf_cache:
            _purge_hf_cache(name)
        # Checkpoint after EVERY model. On Kaggle a 12h timeout hard-kills the
        # session, so if we only wrote at the end a timeout would lose everything.
        # Writing here means every finished model is already on disk to download.
        with open(ckpt, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[ckpt] {i}/{len(cfg.models)} models saved -> {ckpt}")
    _print_summary(all_results)
    return all_results


def _print_summary(results):
    print("\n================ SUMMARY ================")
    print(f"{'model':32s} {'STS rho':>8s} {'STS regret':>11s} {'last-gap':>9s}")
    for r in results:
        s = r.get("sts", {})
        print(f"{r['model']:32s} {s.get('rank_agreement', float('nan')):8.3f}"
              f" {s.get('regret', float('nan')):11.4f}"
              f" {s.get('last_layer_gap', float('nan')):9.4f}")


# --------------------------------------------------------------------------- #
# Human-readable CSV summaries (in addition to the full JSON)
# --------------------------------------------------------------------------- #
def _spear(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(a, float), np.asarray(b, float)).correlation
    return round(float(r), 3) if r == r else ""


def _crit_stats(sig, per_layer, minimize=True):
    """For a signal (lower=better if minimize), return (pick, oracle, regret,
    last_layer_gap, rank_agreement)."""
    s = np.asarray(sig, float)
    crit = -s if minimize else s
    pred = int(np.argmax(crit))
    pl = np.asarray(per_layer, float)
    return (pred, int(pl.argmax()), round(float(pl.max() - pl[pred]), 4),
            round(float(pl.max() - pl[-1]), 4), _spear(crit, pl))


def _tasks_of(r):
    tasks = {}
    s = r.get("sts")
    if isinstance(s, dict) and "per_layer" in s:
        tasks["STS"] = s["per_layer"]
    ret = r.get("retrieval")
    if isinstance(ret, dict) and "per_layer" in ret:
        tasks["RETRIEVAL"] = ret["per_layer"]
    for t, tv in r.get("classification", {}).items():
        if isinstance(tv, dict) and "per_layer" in tv:
            tasks[t] = tv["per_layer"]
    return tasks


def write_summary_csv(results, path):
    """Flat, spreadsheet-friendly CSV: one row per (model, task). For each working
    criterion (intrinsic dimension, curvature) it lists the selected layer, the
    oracle-best layer, the last-layer gap, selection regret, and rank agreement
    (Spearman) with true downstream quality."""
    cols = ["model", "task", "n_layers", "oracle_layer", "last_layer_gap",
            "ID_pick", "ID_regret", "ID_rank_agree",
            "curv_pick", "curv_regret", "curv_rank_agree"]
    rows = []
    for r in results:
        c = r.get("components", {})
        idim, curv = c.get("intrinsic_dimension"), c.get("curvature")
        for t, pl in _tasks_of(r).items():
            row = {"model": r["model"], "task": t, "n_layers": r.get("n_layers", "")}
            if idim:
                p, o, reg, lg, rho = _crit_stats(idim, pl)
                row.update(oracle_layer=o, last_layer_gap=lg,
                           ID_pick=p, ID_regret=reg, ID_rank_agree=rho)
            if curv:
                p, o, reg, lg, rho = _crit_stats(curv, pl)
                row.update(curv_pick=p, curv_regret=reg, curv_rank_agree=rho)
            rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})
    print(f"[csv] wrote {len(rows)} rows -> {path}")


def _layeriq_g(c):
    """The v1 geometry-only criterion (effective rank x isotropy x stability,
    min-max normalised, geometric mean) -- the one that FAILS."""
    def mm(x):
        x = np.asarray(x, float)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)
    return np.cbrt((mm(c["effective_rank"]) + 1e-6)
                   * (mm(c["isotropy"]) + 1e-6) * (mm(c["stability"]) + 1e-6))


def write_failure_csv(results, path):
    """Reproduce the geometry-only FAILURE (paper Sec. 5): for each model x task,
    the layer the v1 criterion selects and its rank agreement with downstream
    quality. The selected layer typically collapses to 0 and rank agreement is
    ~0 or negative -- that is the failure."""
    cols = ["model", "task", "n_layers", "v1_pick", "oracle_layer",
            "v1_regret", "v1_rank_agree"]
    rows = []
    for r in results:
        c = r.get("components", {})
        if "effective_rank" not in c:
            continue
        g = _layeriq_g(c)
        pred = int(np.argmax(g))
        for t, pl in _tasks_of(r).items():
            plv = np.asarray(pl, float)
            rows.append({"model": r["model"], "task": t,
                         "n_layers": r.get("n_layers", ""), "v1_pick": pred,
                         "oracle_layer": int(plv.argmax()),
                         "v1_regret": round(float(plv.max() - plv[pred]), 4),
                         "v1_rank_agree": _spear(g, pl)})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[csv] geometry-only FAILURE summary -> {path}")
    print(f"\n{'model':28s} {'v1_pick':>7s} {'oracle':>7s} {'task':>5s} {'rank_agree':>10s}")
    for row in rows:
        print(f"{row['model'][:28]:28s} {row['v1_pick']:7d} {row['oracle_layer']:7d} "
              f"{row['task']:>5s} {str(row['v1_rank_agree']):>10s}")


if __name__ == "__main__":
    main()
