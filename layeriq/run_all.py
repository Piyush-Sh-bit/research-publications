# =====================================================================
# LayerIQ -- single reproducible run (all tiers)
#
# ONE notebook that runs every experiment in the paper, in order, and writes
# both JSON (full per-layer panel) and readable CSV summaries. Anyone can paste
# this into a Kaggle notebook, Run All, and reproduce every number.
#
# HOW TO USE
#   1. Upload the sibling folder  final_version/dataset/  as a Kaggle DATASET.
#   2. New Kaggle Notebook -> Accelerator = GPU (T4/P100), Internet = ON.
#   3. Add Input -> attach the dataset.
#   4. Paste this whole file into one cell, Run All.
#
# Each tier is an independent block in RUNS below -- comment out any you want to
# skip (e.g. leave only "small" for a ~30-min reproduction). The "high" tier
# (7B, 4-bit) is the slow one; the full sweep is long, so budget accordingly.
#
# Every model run also evaluates the downstream retrieval APPLICATION (paraphrase
# search) in addition to STS + probing-classification.
# =====================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.40", "datasets>=2.18", "scikit-learn",
                "scipy", "bitsandbytes", "accelerate"], check=False)

import os, glob, json
_hits = glob.glob("/kaggle/input/**/config.py", recursive=True) + \
        glob.glob("/kaggle/working/**/config.py", recursive=True)
assert _hits, "Code not found -- Add Input (attach the dataset) and check the .py files uploaded."
SRC = os.path.dirname(_hits[0]); sys.path.insert(0, SRC)
print("Using code from:", SRC)

from config import (ExperimentConfig, TIER_SMALL, TIER_MEDIUM,
                    TIER_ENCODERS, TIER_LARGE)
import run_experiment as R

OUT = "/kaggle/working/results"; os.makedirs(OUT, exist_ok=True)

# (tag, models, load_in_4bit).
# Measured on a Kaggle T4: the whole sweep below is ~4h at full probe size, so it
# fits in one 12h session. Results are checkpointed after every model, so even an
# interrupted run leaves every finished model on disk in results.json.
RUNS = [
    ("small",  list(TIER_SMALL) + list(TIER_ENCODERS), False),   # ~45 min (16 small models)
    ("medium", list(TIER_MEDIUM),                      False),   # ~1.5 h  (8 models, 1-3B)
    ("high1",  list(TIER_LARGE)[:2],                   True),    # ~45 min (2x 7B, 4-bit)
    ("high2",  list(TIER_LARGE)[2:],                   True),    # ~30 min (2x 7B, 4-bit)
]

for tag, models, fourbit in RUNS:
    print(f"\n########################  TIER: {tag}  ({len(models)} models)  "
          f"########################")
    cfg = ExperimentConfig(models=models, n_unlabeled=512, out_dir=OUT,
                           load_in_4bit=fourbit)
    results = R.main(cfg)
    json.dump(results, open(f"{OUT}/results_{tag}.json", "w"), indent=2)
    R.write_summary_csv(results, f"{OUT}/results_{tag}_summary.csv")
    if tag == "small":
        # the geometry-only FAILURE (paper Sec. 5), derived from the small tier
        R.write_failure_csv(results, f"{OUT}/results_failure_geometry_v1.csv")

