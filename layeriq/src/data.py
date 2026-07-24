"""Dataset loading. Uses HuggingFace `datasets`; all small enough for Kaggle.

- Unlabeled corpus: sentences from WikiText (NO labels used).
- STS-B: sentence pairs + gold similarity (validation split) -> Spearman.
- SST-2 / TREC: sentence + label -> linear-probe accuracy.
"""
from __future__ import annotations

from typing import List, Tuple

from datasets import load_dataset


def load_unlabeled(name: str, subset: str, n: int, seed: int = 0) -> List[str]:
    ds = load_dataset(name, subset, split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 40]
    rng = __import__("random").Random(seed)
    rng.shuffle(texts)
    return texts[:n]


def load_sts() -> Tuple[List[str], List[str], List[float]]:
    ds = load_dataset("glue", "stsb", split="validation")
    return list(ds["sentence1"]), list(ds["sentence2"]), [float(x) for x in ds["label"]]


def load_retrieval(task: str = "mrpc", limit: int = 500) -> Tuple[List[str], List[str]]:
    """Paraphrase retrieval: return (queries, pool) where query i's correct match is
    pool i. We use only positive (paraphrase) pairs, so retrieval accuracy@1 over the
    pool is well defined and label-free at test time."""
    if task == "mrpc":
        ds = load_dataset("glue", "mrpc", split="train")
        s1, s2, lab = ds["sentence1"], ds["sentence2"], ds["label"]
    elif task == "paws":
        ds = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
        s1, s2, lab = ds["sentence1"], ds["sentence2"], ds["label"]
    else:
        raise ValueError(f"unknown retrieval task {task}")
    q = [a for a, l in zip(s1, lab) if int(l) == 1][:limit]
    p = [b for b, l in zip(s2, lab) if int(l) == 1][:limit]
    return q, p


def load_classification(task: str, n_train: int = 2000, n_test: int = 1000):
    """Return (train_texts, train_y, test_texts, test_y)."""
    if task == "sst2":
        ds = load_dataset("glue", "sst2")
        tr, te = ds["train"], ds["validation"]
        return (
            list(tr["sentence"])[:n_train], list(tr["label"])[:n_train],
            list(te["sentence"])[:n_test], list(te["label"])[:n_test],
        )
    if task == "trec":
        ds = _load_trec()
        tr, te = ds["train"], ds["test"]
        text_key = "text" if "text" in tr.column_names else "sentence"
        label_key = next(
            (k for k in ("coarse_label", "label-coarse", "label_coarse", "label")
             if k in tr.column_names),
            "label",
        )
        return (
            list(tr[text_key])[:n_train], list(tr[label_key])[:n_train],
            list(te[text_key])[:n_test], list(te[label_key])[:n_test],
        )
    # SentEval-style probing tasks (SetFit mirrors are parquet -> no script needed).
    setfit = {"subj": "SetFit/subj", "mr": "SetFit/mr",
              "cr": "SetFit/SentEval-CR", "sst5": "SetFit/sst5"}
    if task in setfit:
        ds = load_dataset(setfit[task])
        tr = ds["train"]
        te = ds["test"] if "test" in ds else ds["validation"]
        tkey = "text" if "text" in tr.column_names else "sentence"
        lkey = "label" if "label" in tr.column_names else (
            "label_text" if "label_text" in tr.column_names else tr.column_names[-1])
        return (
            list(tr[tkey])[:n_train], list(tr[lkey])[:n_train],
            list(te[tkey])[:n_test], list(te[lkey])[:n_test],
        )
    raise ValueError(f"unknown task {task}")


def _load_trec():
    """Load TREC without a dataset loading script.

    `datasets` >= 3/4 removed script-based loaders, so `load_dataset("trec")`
    (which ships a `trec.py` script) now raises. We load TREC from its parquet
    export instead, falling back to the legacy scripted path only on old
    `datasets`. Raises only if every source fails.
    """
    errs = []
    # Discover the actual parquet files on the auto-convert branch (filename
    # layout varies across HF exports, so we glob rather than hardcode).
    try:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        for repo in ("trec", "CogComp/trec"):
            base = f"datasets/{repo}@refs/convert/parquet"
            try:
                train = (fs.glob(f"{base}/**/train/*.parquet")
                         or fs.glob(f"{base}/**/*train*.parquet"))
                test = (fs.glob(f"{base}/**/test/*.parquet")
                        or fs.glob(f"{base}/**/*test*.parquet"))
                if train and test:
                    return load_dataset("parquet", data_files={
                        "train": ["hf://" + p for p in train],
                        "test":  ["hf://" + p for p in test],
                    })
            except Exception as e:  # noqa: BLE001
                errs.append(f"{repo}: {str(e).splitlines()[0]}")
    except Exception as e:  # noqa: BLE001
        errs.append(f"hffs: {str(e).splitlines()[0]}")
    try:
        return load_dataset("trec")            # legacy: only works on datasets < 3
    except Exception as e:  # noqa: BLE001
        errs.append(str(e).splitlines()[0])
    raise RuntimeError("TREC unavailable script-free: " + " | ".join(errs))
