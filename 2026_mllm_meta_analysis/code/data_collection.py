"""
data_collection.py
==================
Loads the verified benchmark table used throughout the analysis.

The dataset is held in `data/benchmark_scores_verified.csv` rather than hard-coded
here, so that the exact values analysed can be inspected and cited directly. Each
row is one model-benchmark observation and carries its own provenance:

  model, benchmark, score          the observation
  params_b, vision_encoder,        model metadata
  llm_backbone, training_strategy,
  year
  source, source_url,              where the MODEL is described, its venue, and
  source_venue, peer_reviewed,     whether that publication was peer reviewed
  manuscript_ref
  score_source, score_source_url,  where THIS SCORE was taken from, and whether
  score_source_type                that was the model paper, the benchmark paper,
                                   a leaderboard, or another paper's comparison
                                   table

Two conventions matter for comparability:

* MME values are MME-Perception (max 2000) throughout. Perception + cognition
  sums (max 2800) are not comparable with perception scores, so where only a sum
  was available the cell was left blank rather than mixed in.
* SEED-Bench and SEED-Bench-Image are kept as separate benchmarks, because the
  published figures are not interchangeable.

Scores that could not be traced to a defensible source are blank, never imputed.
Models with undisclosed parameter counts (GPT-4V, Gemini-Pro-V) are retained in
the table but fall out of the modelling sample, since model scale is the primary
moderator; `get_benchmark_data()` returns the modelling sample and
`get_full_table()` returns everything.
"""

import os

import numpy as np
import pandas as pd

_DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                         "benchmark_scores_verified.csv")

# Simplified vision-encoder groupings used for the encoder-family moderator.
_ENCODER_MAP = {
    "CLIP-ViT-L": "CLIP-family",
    "CLIP-ViT-H": "CLIP-family",
    "EVA-ViT-G": "EVA-family",
    "EVA2-CLIP-E": "EVA-family",
    "ViT-bigG": "ViT-large",
    "ViT-L": "ViT-large",
    "ViT-BigHuge": "ViT-large",
    "InternViT-6B": "InternViT",
    "SigLIP-L": "SigLIP",
    "SigLIP-L + SAM-B": "SigLIP",
    "linear_projection": "linear_projection",
    "multi_encoder": "multi_encoder",
    "proprietary": "proprietary",
}


def get_full_table() -> pd.DataFrame:
    """Every row of the verified table, including blank scores and the models
    whose parameter counts are undisclosed. Use for provenance reporting."""
    return pd.read_csv(os.path.normpath(_DATA_CSV))


def get_benchmark_data() -> pd.DataFrame:
    """The modelling sample: rows with both a sourced score and a known
    parameter count, plus the derived columns the analyses expect."""
    df = get_full_table()
    df = df.dropna(subset=["score", "params_b"]).copy()

    df["log_params"] = np.log10(df["params_b"])
    df["scale_category"] = pd.cut(
        df["params_b"],
        bins=[0, 10, 50, 100, 2000],
        labels=["Small (<10B)", "Medium (10-50B)", "Large (50-100B)", "Very Large (>100B)"],
    )
    df["encoder_family"] = df["vision_encoder"].map(_ENCODER_MAP).fillna(df["vision_encoder"])
    return df.reset_index(drop=True)


def get_benchmark_metadata() -> dict:
    """Metadata about each benchmark (scale, task type, etc.)."""
    return {
        "MMBench": {
            "full_name": "MMBench",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "comprehensive",
            "description": "Multi-ability benchmark covering perception, reasoning, and knowledge",
        },
        "SEED-Bench": {
            "full_name": "SEED-Bench",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "comprehensive",
            "description": "Spatial and temporal understanding in image and video",
        },
        "SEED-Bench-Image": {
            "full_name": "SEED-Bench (image split)",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "comprehensive",
            "description": "Image-only split of SEED-Bench; not interchangeable with the full benchmark",
        },
        "MM-Vet": {
            "full_name": "MM-Vet",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "open_ended",
            "description": "Open-ended visual chat evaluation using GPT-4 scoring",
        },
        "MME": {
            "full_name": "MME (Perception)",
            "scale": "raw_score",
            "max_score": 2000.0,
            "task_type": "comprehensive",
            "description": "Perception abilities via yes/no questions; perception split only",
        },
        "TextVQA": {
            "full_name": "TextVQA",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "ocr",
            "description": "Visual question answering requiring text reading in images",
        },
        "POPE": {
            "full_name": "POPE",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "hallucination",
            "description": "Polling-based object probing for hallucination evaluation",
        },
        "VQAv2": {
            "full_name": "VQAv2",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "vqa",
            "description": "General visual question answering on natural images",
        },
    }


if __name__ == "__main__":
    full = get_full_table()
    df = get_benchmark_data()
    print("full table:      %d rows, %d models, %d with a score"
          % (len(full), full["model"].nunique(), full["score"].notna().sum()))
    print("modelling sample:%d rows, %d models, %d benchmarks"
          % (len(df), df["model"].nunique(), df["benchmark"].nunique()))
    excluded = sorted(set(full["model"]) - set(df["model"]))
    print("excluded (no disclosed parameter count):", excluded)
    print("\nrecords per benchmark:")
    print(df.groupby("benchmark").size().sort_values(ascending=False).to_string())
