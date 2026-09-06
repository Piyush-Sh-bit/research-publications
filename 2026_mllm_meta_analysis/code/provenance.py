"""
provenance.py
=============
Per-record source information for the benchmark values analysed in this study.

This module is purely descriptive. It adds provenance columns to the data
provenance table (Table 13) and changes no score, no model metadata and no
analysis result.

Two things are recorded for every observation:

  source / source_url / source_venue / peer_reviewed
      where the MODEL is described

  score_source / score_source_url / score_source_type
      the document the SCORE itself was read from

`score_source_type` is one of:
  model_paper          the model's own publication reported this benchmark
  benchmark_paper      the benchmark's own paper reported this model
  leaderboard          a public leaderboard reported this model
  original_compilation the value comes from the study's original data
                       collection and could not be re-traced to a single
                       citable document; retained and labelled rather than
                       silently attributed
"""

import pandas as pd

# --------------------------------------------------------------------------
# Where each model is described: key -> (webpage, venue, peer reviewed)
# --------------------------------------------------------------------------
MODEL_SOURCES = {
    "liu2024improved":     ("https://arxiv.org/abs/2310.03744", "CVPR 2024", "yes"),
    "dai2023instructblip": ("https://arxiv.org/abs/2305.06500", "NeurIPS 2023", "yes"),
    "li2023blip2":         ("https://arxiv.org/abs/2301.12597", "ICML 2023", "yes"),
    "bai2023qwenvl":       ("https://arxiv.org/abs/2308.12966", "arXiv preprint", "no"),
    "tong2024cambrian":    ("https://arxiv.org/abs/2406.16860", "NeurIPS 2024 (Oral)", "yes"),
    "openai2023gpt4v":     ("https://arxiv.org/abs/2309.17421", "OpenAI system card", "no"),
    "team2023gemini":      ("https://arxiv.org/abs/2312.11805", "Google technical report", "no"),
    "zhu2023minigpt4":     ("https://arxiv.org/abs/2304.10592", "ICLR 2024", "yes"),
    "ye2024mplugowl2":     ("https://arxiv.org/abs/2311.04257", "CVPR 2024", "yes"),
    "wang2024cogvlm":      ("https://arxiv.org/abs/2311.03079", "NeurIPS 2024", "yes"),
    "chen2024sharegpt4v":  ("https://arxiv.org/abs/2311.12793", "ECCV 2024", "yes"),
    "li2024monkey":        ("https://arxiv.org/abs/2311.06607", "CVPR 2024", "yes"),
    "chen2024internvl":    ("https://arxiv.org/abs/2404.16821", "Science China Information Sciences 2024", "yes"),
    "lu2024deepseekvl":    ("https://arxiv.org/abs/2403.05525", "arXiv technical report", "no"),
    "young2024yi":         ("https://arxiv.org/abs/2403.04652", "arXiv technical report", "no"),
    "bavishi2023fuyu":     ("https://www.adept.ai/blog/fuyu-8b/", "Adept blog post", "no"),
    "liu2024llavanext":    ("https://llava-vl.github.io/blog/2024-01-30-llava-next/", "LLaVA project blog post", "no"),
}

_MMVET = ("yu2024mmvet", "https://arxiv.org/abs/2308.02490", "benchmark_paper")
_SEED = ("li2024seedbench", "https://arxiv.org/abs/2307.16125", "benchmark_paper")
_MME_LB = ("fu2023mme_leaderboard",
           "https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation",
           "leaderboard")
_UNTRACED = ("original_compilation", "", "original_compilation")

# --------------------------------------------------------------------------
# Per-value overrides, established by checking the printed value against the
# named document. Any (model, benchmark) pair not listed here defaults to the
# model's own publication.
# --------------------------------------------------------------------------
SCORE_SOURCES = {
    # located in the MM-Vet paper's own results table
    ("BLIP-2", "MM-Vet"): _MMVET,
    ("MiniGPT-4", "MM-Vet"): _MMVET,
    ("InstructBLIP-7B", "MM-Vet"): _MMVET,
    ("InstructBLIP-13B", "MM-Vet"): _MMVET,

    # located in the SEED-Bench paper's own results table
    ("BLIP-2", "SEED-Bench"): _SEED,
    ("InstructBLIP-7B", "SEED-Bench"): _SEED,

    # located on the MME leaderboard (the MME paper prints only subtask scores)
    ("BLIP-2", "MME"): _MME_LB,
    ("InstructBLIP-7B", "MME"): _MME_LB,
    ("Qwen-VL-Chat", "MME"): _MME_LB,
    ("mPLUG-Owl2", "MME"): _MME_LB,

    # values retained from the study's original data collection that could not
    # be re-traced to one citable document
    ("GPT-4V", "MMBench"): _UNTRACED,
    ("GPT-4V", "SEED-Bench"): _UNTRACED,
    ("GPT-4V", "MM-Vet"): _UNTRACED,
    ("GPT-4V", "MME"): _UNTRACED,
    ("GPT-4V", "TextVQA"): _UNTRACED,
    ("MiniGPT-4", "MMBench"): _UNTRACED,
    ("MiniGPT-4", "MME"): _UNTRACED,
    ("BLIP-2", "MMBench"): _UNTRACED,
    ("InstructBLIP-7B", "MMBench"): _UNTRACED,
    ("InstructBLIP-13B", "MMBench"): _UNTRACED,
    ("InstructBLIP-13B", "MME"): _UNTRACED,
    ("Cambrian-1-8B", "MME"): _UNTRACED,
    ("InternVL-Chat-V1.5", "MME"): _UNTRACED,
    ("Yi-VL-34B", "MME"): _UNTRACED,
}


def add_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with provenance columns appended.

    Scores, model metadata and row order are untouched.
    """
    out = df.copy()
    blank = ("", "", "")
    out["source_url"] = out["source"].map(lambda s: MODEL_SOURCES.get(s, blank)[0])
    out["source_venue"] = out["source"].map(lambda s: MODEL_SOURCES.get(s, blank)[1])
    out["peer_reviewed"] = out["source"].map(lambda s: MODEL_SOURCES.get(s, blank)[2])

    def per_value(row):
        key = (row["model"], row["benchmark"])
        if key in SCORE_SOURCES:
            k, url, kind = SCORE_SOURCES[key]
        else:
            k = row["source"]
            url = MODEL_SOURCES.get(row["source"], blank)[0]
            kind = "model_paper"
        return pd.Series([k, url, kind],
                         index=["score_source", "score_source_url", "score_source_type"])

    return pd.concat([out, out.apply(per_value, axis=1)], axis=1)


def summary(df: pd.DataFrame) -> str:
    """One-line summary used by the pipeline and the data availability text."""
    p = add_provenance(df)
    counts = p["score_source_type"].value_counts().to_dict()
    n_untraced = counts.get("original_compilation", 0)
    return ("%d records; %d with a citable source and link, %d retained from the "
            "original data collection. Breakdown: %s"
            % (len(p), len(p) - n_untraced, n_untraced,
               ", ".join("%s=%d" % kv for kv in sorted(counts.items()))))


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.stdout.reconfigure(encoding="utf-8")
    from data_collection import get_benchmark_data
    print(summary(get_benchmark_data()))
