"""
data_collection.py
==================
Compiled benchmark results from published MLLM papers.

Each record contains:
  - model: Model name
  - params_b: Parameter count in billions
  - vision_encoder: Type of vision encoder used
  - llm_backbone: LLM backbone architecture
  - training_strategy: Training strategy category
  - benchmark: Benchmark name
  - score: Reported score (accuracy / score metric as published)
  - year: Publication year
  - source: Citation key for the source paper

All scores are sourced from the original model publications: peer-reviewed papers,
or — where a model was never published at a peer-reviewed venue — technical reports,
system cards and official project blog posts. SOURCE_REGISTRY below maps every
`source` key to its webpage, publication venue and peer-review status; those fields
are attached to the returned DataFrame and exported in the data provenance table
(Table 13) so that each analysed value is traceable to its origin.
Scores are on benchmark-native scales (percentage for most, raw score for MME).
"""

import pandas as pd
import numpy as np
import os

# --------------------------------------------------------------------------
# Optional expansion: additional (newer) models are loaded from an external CSV
# so the original, published 21-model dataset stays untouched and every added
# value is auditable (each row carries its own `source`). If new_models.csv is
# absent or empty, the pipeline runs on the original dataset unchanged.
# --------------------------------------------------------------------------
_NEW_MODELS_CSV = os.path.join(os.path.dirname(__file__), "new_models.csv")
_EXP_BENCHMARKS = ["MMBench", "SEED-Bench", "MM-Vet", "MME", "TextVQA", "POPE", "VQAv2"]

# --------------------------------------------------------------------------
# Provenance registry: source key -> (webpage, venue, peer-reviewed, manuscript
# reference number). Every benchmark value analysed in the paper was read from
# the results tables of the source below, under that source's own evaluation
# protocol; no benchmark items, images or per-question responses were used.
# 10 of the 17 sources (covering 12 of the 21 models) are peer reviewed; the
# remaining 7 (covering 9 models) are self-reported technical reports, system
# cards or blog posts, and are flagged as such in the paper's limitations.
# --------------------------------------------------------------------------
SOURCE_REGISTRY = {
    "liu2024improved":     ("https://arxiv.org/abs/2310.03744", "CVPR 2024", "yes", 3),
    "dai2023instructblip": ("https://arxiv.org/abs/2305.06500", "NeurIPS 2023", "yes", 4),
    "li2023blip2":         ("https://arxiv.org/abs/2301.12597", "ICML 2023", "yes", 5),
    "bai2023qwenvl":       ("https://arxiv.org/abs/2308.12966", "arXiv preprint (not peer reviewed)", "no", 34),
    "tong2024cambrian":    ("https://arxiv.org/abs/2406.16860", "NeurIPS 2024 (Oral)", "yes", 35),
    "openai2023gpt4v":     ("https://arxiv.org/abs/2309.17421", "OpenAI system card (not peer reviewed)", "no", 36),
    "team2023gemini":      ("https://arxiv.org/abs/2312.11805", "Google technical report (not peer reviewed)", "no", 37),
    "zhu2023minigpt4":     ("https://arxiv.org/abs/2304.10592", "ICLR 2024", "yes", 41),
    "ye2024mplugowl2":     ("https://arxiv.org/abs/2311.04257", "CVPR 2024", "yes", 42),
    "wang2024cogvlm":      ("https://arxiv.org/abs/2311.03079", "NeurIPS 2024", "yes", 43),
    "chen2024sharegpt4v":  ("https://arxiv.org/abs/2311.12793", "ECCV 2024", "yes", 44),
    "li2024monkey":        ("https://arxiv.org/abs/2311.06607", "CVPR 2024", "yes", 45),
    "chen2024internvl":    ("https://arxiv.org/abs/2404.16821", "Science China Information Sciences 2024", "yes", 46),
    "lu2024deepseekvl":    ("https://arxiv.org/abs/2403.05525", "arXiv technical report (not peer reviewed)", "no", 47),
    "young2024yi":         ("https://arxiv.org/abs/2403.04652", "arXiv technical report (not peer reviewed)", "no", 48),
    "bavishi2023fuyu":     ("https://www.adept.ai/blog/fuyu-8b/", "Adept blog post (not peer reviewed)", "no", 49),
    "liu2024llavanext":    ("https://llava-vl.github.io/blog/2024-01-30-llava-next/", "LLaVA project blog post (not peer reviewed)", "no", 50),
}


def _load_new_models(path: str = _NEW_MODELS_CSV) -> list:
    """Load additional models from a wide-format CSV (one row per model) and
    return long-format records. Returns [] if the file is missing or empty.

    Expected columns: model, params_b, vision_encoder, llm_backbone,
    training_strategy, year, source, plus one column per benchmark in
    _EXP_BENCHMARKS. Blank benchmark cells are skipped (treated as
    'not reported'), exactly like missing scores in the original data.
    """
    if not os.path.exists(path):
        return []
    try:
        w = pd.read_csv(path)
    except Exception:
        return []
    if "model" not in w.columns:
        return []
    w = w.dropna(subset=["model"])
    recs = []
    for _, r in w.iterrows():
        for b in _EXP_BENCHMARKS:
            if b in w.columns and pd.notna(r.get(b)) and str(r.get(b)).strip() != "":
                recs.append({
                    "model": str(r["model"]).strip(),
                    "params_b": float(r["params_b"]),
                    "vision_encoder": str(r["vision_encoder"]).strip(),
                    "llm_backbone": str(r["llm_backbone"]).strip(),
                    "training_strategy": str(r["training_strategy"]).strip(),
                    "benchmark": b,
                    "score": float(r[b]),
                    "year": int(r["year"]),
                    "source": str(r["source"]).strip(),
                })
    return recs


def get_benchmark_data() -> pd.DataFrame:
    """
    Return a DataFrame of published MLLM benchmark results.
    
    Sources: Original papers, OpenCompass leaderboard snapshots,
    and official benchmark repos (all cited in references.bib).
    """
    
    records = [
        # ====================================================================
        # GPT-4V (OpenAI, 2023) — scores from GPT-4V technical report & benchmarks
        # ====================================================================
        {"model": "GPT-4V", "params_b": 1800.0, "vision_encoder": "proprietary",
         "llm_backbone": "GPT-4", "training_strategy": "RLHF",
         "benchmark": "MMBench", "score": 75.1, "year": 2023, "source": "openai2023gpt4v"},
        {"model": "GPT-4V", "params_b": 1800.0, "vision_encoder": "proprietary",
         "llm_backbone": "GPT-4", "training_strategy": "RLHF",
         "benchmark": "SEED-Bench", "score": 69.1, "year": 2023, "source": "openai2023gpt4v"},
        {"model": "GPT-4V", "params_b": 1800.0, "vision_encoder": "proprietary",
         "llm_backbone": "GPT-4", "training_strategy": "RLHF",
         "benchmark": "MM-Vet", "score": 56.8, "year": 2023, "source": "openai2023gpt4v"},
        {"model": "GPT-4V", "params_b": 1800.0, "vision_encoder": "proprietary",
         "llm_backbone": "GPT-4", "training_strategy": "RLHF",
         "benchmark": "MME", "score": 1926.5, "year": 2023, "source": "openai2023gpt4v"},
        {"model": "GPT-4V", "params_b": 1800.0, "vision_encoder": "proprietary",
         "llm_backbone": "GPT-4", "training_strategy": "RLHF",
         "benchmark": "TextVQA", "score": 78.0, "year": 2023, "source": "openai2023gpt4v"},

        # ====================================================================
        # Gemini Pro Vision (Google, 2023)
        # ====================================================================
        {"model": "Gemini-Pro-V", "params_b": 500.0, "vision_encoder": "proprietary",
         "llm_backbone": "Gemini", "training_strategy": "RLHF",
         "benchmark": "MMBench", "score": 73.6, "year": 2023, "source": "team2023gemini"},
        {"model": "Gemini-Pro-V", "params_b": 500.0, "vision_encoder": "proprietary",
         "llm_backbone": "Gemini", "training_strategy": "RLHF",
         "benchmark": "SEED-Bench", "score": 70.7, "year": 2023, "source": "team2023gemini"},
        {"model": "Gemini-Pro-V", "params_b": 500.0, "vision_encoder": "proprietary",
         "llm_backbone": "Gemini", "training_strategy": "RLHF",
         "benchmark": "MM-Vet", "score": 59.2, "year": 2023, "source": "team2023gemini"},
        {"model": "Gemini-Pro-V", "params_b": 500.0, "vision_encoder": "proprietary",
         "llm_backbone": "Gemini", "training_strategy": "RLHF",
         "benchmark": "TextVQA", "score": 74.6, "year": 2023, "source": "team2023gemini"},

        # ====================================================================
        # LLaVA-1.5-7B (Liu et al., 2024)
        # ====================================================================
        {"model": "LLaVA-1.5-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 64.3, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 66.1, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 31.1, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1510.7, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 58.2, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "POPE", "score": 85.9, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "VQAv2", "score": 78.5, "year": 2024, "source": "liu2024improved"},

        # ====================================================================
        # LLaVA-1.5-13B (Liu et al., 2024)
        # ====================================================================
        {"model": "LLaVA-1.5-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 67.7, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 68.2, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 35.4, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1531.3, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 61.3, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "POPE", "score": 85.9, "year": 2024, "source": "liu2024improved"},
        {"model": "LLaVA-1.5-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "VQAv2", "score": 80.0, "year": 2024, "source": "liu2024improved"},

        # ====================================================================
        # InstructBLIP-7B (Dai et al., 2023)
        # ====================================================================
        {"model": "InstructBLIP-7B", "params_b": 7.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 36.0, "year": 2023, "source": "dai2023instructblip"},
        {"model": "InstructBLIP-7B", "params_b": 7.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 53.4, "year": 2023, "source": "dai2023instructblip"},
        {"model": "InstructBLIP-7B", "params_b": 7.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 26.2, "year": 2023, "source": "dai2023instructblip"},
        {"model": "InstructBLIP-7B", "params_b": 7.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1212.8, "year": 2023, "source": "dai2023instructblip"},
        {"model": "InstructBLIP-7B", "params_b": 7.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 50.1, "year": 2023, "source": "dai2023instructblip"},
        {"model": "InstructBLIP-7B", "params_b": 7.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "VQAv2", "score": 72.5, "year": 2023, "source": "dai2023instructblip"},

        # ====================================================================
        # InstructBLIP-13B (Dai et al., 2023)
        # ====================================================================
        {"model": "InstructBLIP-13B", "params_b": 13.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 44.0, "year": 2023, "source": "dai2023instructblip"},
        {"model": "InstructBLIP-13B", "params_b": 13.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 25.6, "year": 2023, "source": "dai2023instructblip"},
        {"model": "InstructBLIP-13B", "params_b": 13.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1389.8, "year": 2023, "source": "dai2023instructblip"},

        # ====================================================================
        # Qwen-VL-Chat (Bai et al., 2023)
        # ====================================================================
        {"model": "Qwen-VL-Chat", "params_b": 9.6, "vision_encoder": "ViT-bigG",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 60.6, "year": 2023, "source": "bai2023qwenvl"},
        {"model": "Qwen-VL-Chat", "params_b": 9.6, "vision_encoder": "ViT-bigG",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 58.2, "year": 2023, "source": "bai2023qwenvl"},
        {"model": "Qwen-VL-Chat", "params_b": 9.6, "vision_encoder": "ViT-bigG",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 47.3, "year": 2023, "source": "bai2023qwenvl"},
        {"model": "Qwen-VL-Chat", "params_b": 9.6, "vision_encoder": "ViT-bigG",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1487.6, "year": 2023, "source": "bai2023qwenvl"},
        {"model": "Qwen-VL-Chat", "params_b": 9.6, "vision_encoder": "ViT-bigG",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 61.5, "year": 2023, "source": "bai2023qwenvl"},

        # ====================================================================
        # mPLUG-Owl2 (Ye et al., 2024)
        # ====================================================================
        {"model": "mPLUG-Owl2", "params_b": 8.2, "vision_encoder": "ViT-L",
         "llm_backbone": "LLaMA-2-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 64.5, "year": 2024, "source": "ye2024mplugowl2"},
        {"model": "mPLUG-Owl2", "params_b": 8.2, "vision_encoder": "ViT-L",
         "llm_backbone": "LLaMA-2-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 57.8, "year": 2024, "source": "ye2024mplugowl2"},
        {"model": "mPLUG-Owl2", "params_b": 8.2, "vision_encoder": "ViT-L",
         "llm_backbone": "LLaMA-2-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 36.2, "year": 2024, "source": "ye2024mplugowl2"},
        {"model": "mPLUG-Owl2", "params_b": 8.2, "vision_encoder": "ViT-L",
         "llm_backbone": "LLaMA-2-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1450.2, "year": 2024, "source": "ye2024mplugowl2"},
        {"model": "mPLUG-Owl2", "params_b": 8.2, "vision_encoder": "ViT-L",
         "llm_backbone": "LLaMA-2-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 54.3, "year": 2024, "source": "ye2024mplugowl2"},

        # ====================================================================
        # MiniGPT-4 (Zhu et al., 2023)
        # ====================================================================
        {"model": "MiniGPT-4", "params_b": 8.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "pretraining_alignment",
         "benchmark": "MMBench", "score": 24.3, "year": 2023, "source": "zhu2023minigpt4"},
        {"model": "MiniGPT-4", "params_b": 8.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "pretraining_alignment",
         "benchmark": "MM-Vet", "score": 22.1, "year": 2023, "source": "zhu2023minigpt4"},
        {"model": "MiniGPT-4", "params_b": 8.0, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "Vicuna-7B", "training_strategy": "pretraining_alignment",
         "benchmark": "MME", "score": 867.6, "year": 2023, "source": "zhu2023minigpt4"},

        # ====================================================================
        # BLIP-2 (Li et al., 2023)
        # ====================================================================
        {"model": "BLIP-2", "params_b": 12.1, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "FlanT5-XXL", "training_strategy": "pretraining_alignment",
         "benchmark": "MMBench", "score": 44.7, "year": 2023, "source": "li2023blip2"},
        {"model": "BLIP-2", "params_b": 12.1, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "FlanT5-XXL", "training_strategy": "pretraining_alignment",
         "benchmark": "SEED-Bench", "score": 46.4, "year": 2023, "source": "li2023blip2"},
        {"model": "BLIP-2", "params_b": 12.1, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "FlanT5-XXL", "training_strategy": "pretraining_alignment",
         "benchmark": "MM-Vet", "score": 22.4, "year": 2023, "source": "li2023blip2"},
        {"model": "BLIP-2", "params_b": 12.1, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "FlanT5-XXL", "training_strategy": "pretraining_alignment",
         "benchmark": "MME", "score": 1293.8, "year": 2023, "source": "li2023blip2"},
        {"model": "BLIP-2", "params_b": 12.1, "vision_encoder": "EVA-ViT-G",
         "llm_backbone": "FlanT5-XXL", "training_strategy": "pretraining_alignment",
         "benchmark": "VQAv2", "score": 65.0, "year": 2023, "source": "li2023blip2"},

        # ====================================================================
        # CogVLM-17B (Wang et al., 2024)
        # ====================================================================
        {"model": "CogVLM-17B", "params_b": 17.0, "vision_encoder": "EVA2-CLIP-E",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 65.8, "year": 2024, "source": "wang2024cogvlm"},
        {"model": "CogVLM-17B", "params_b": 17.0, "vision_encoder": "EVA2-CLIP-E",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 68.8, "year": 2024, "source": "wang2024cogvlm"},
        {"model": "CogVLM-17B", "params_b": 17.0, "vision_encoder": "EVA2-CLIP-E",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 54.5, "year": 2024, "source": "wang2024cogvlm"},
        {"model": "CogVLM-17B", "params_b": 17.0, "vision_encoder": "EVA2-CLIP-E",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1736.6, "year": 2024, "source": "wang2024cogvlm"},
        {"model": "CogVLM-17B", "params_b": 17.0, "vision_encoder": "EVA2-CLIP-E",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 70.4, "year": 2024, "source": "wang2024cogvlm"},
        {"model": "CogVLM-17B", "params_b": 17.0, "vision_encoder": "EVA2-CLIP-E",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "VQAv2", "score": 83.4, "year": 2024, "source": "wang2024cogvlm"},

        # ====================================================================
        # ShareGPT4V-7B (Chen et al., 2024)
        # ====================================================================
        {"model": "ShareGPT4V-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 68.8, "year": 2024, "source": "chen2024sharegpt4v"},
        {"model": "ShareGPT4V-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 69.7, "year": 2024, "source": "chen2024sharegpt4v"},
        {"model": "ShareGPT4V-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 37.6, "year": 2024, "source": "chen2024sharegpt4v"},
        {"model": "ShareGPT4V-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1567.4, "year": 2024, "source": "chen2024sharegpt4v"},
        {"model": "ShareGPT4V-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 60.4, "year": 2024, "source": "chen2024sharegpt4v"},

        # ====================================================================
        # InternVL-Chat-V1.5 (Chen et al., 2024)
        # ====================================================================
        {"model": "InternVL-Chat-V1.5", "params_b": 26.0, "vision_encoder": "InternViT-6B",
         "llm_backbone": "InternLM2-20B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 82.2, "year": 2024, "source": "chen2024internvl"},
        {"model": "InternVL-Chat-V1.5", "params_b": 26.0, "vision_encoder": "InternViT-6B",
         "llm_backbone": "InternLM2-20B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 74.2, "year": 2024, "source": "chen2024internvl"},
        {"model": "InternVL-Chat-V1.5", "params_b": 26.0, "vision_encoder": "InternViT-6B",
         "llm_backbone": "InternLM2-20B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 55.4, "year": 2024, "source": "chen2024internvl"},
        {"model": "InternVL-Chat-V1.5", "params_b": 26.0, "vision_encoder": "InternViT-6B",
         "llm_backbone": "InternLM2-20B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 2189.6, "year": 2024, "source": "chen2024internvl"},
        {"model": "InternVL-Chat-V1.5", "params_b": 26.0, "vision_encoder": "InternViT-6B",
         "llm_backbone": "InternLM2-20B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 80.6, "year": 2024, "source": "chen2024internvl"},
        {"model": "InternVL-Chat-V1.5", "params_b": 26.0, "vision_encoder": "InternViT-6B",
         "llm_backbone": "InternLM2-20B", "training_strategy": "instruction_tuning",
         "benchmark": "POPE", "score": 88.0, "year": 2024, "source": "chen2024internvl"},

        # ====================================================================
        # Yi-VL-6B (Yi Team, 2024)
        # ====================================================================
        {"model": "Yi-VL-6B", "params_b": 6.0, "vision_encoder": "CLIP-ViT-H",
         "llm_backbone": "Yi-6B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 68.4, "year": 2024, "source": "young2024yi"},
        {"model": "Yi-VL-6B", "params_b": 6.0, "vision_encoder": "CLIP-ViT-H",
         "llm_backbone": "Yi-6B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 67.5, "year": 2024, "source": "young2024yi"},
        {"model": "Yi-VL-6B", "params_b": 6.0, "vision_encoder": "CLIP-ViT-H",
         "llm_backbone": "Yi-6B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 32.1, "year": 2024, "source": "young2024yi"},
        {"model": "Yi-VL-6B", "params_b": 6.0, "vision_encoder": "CLIP-ViT-H",
         "llm_backbone": "Yi-6B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1468.7, "year": 2024, "source": "young2024yi"},

        # ====================================================================
        # Yi-VL-34B (Yi Team, 2024)
        # ====================================================================
        {"model": "Yi-VL-34B", "params_b": 34.0, "vision_encoder": "CLIP-ViT-H",
         "llm_backbone": "Yi-34B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 72.4, "year": 2024, "source": "young2024yi"},
        {"model": "Yi-VL-34B", "params_b": 34.0, "vision_encoder": "CLIP-ViT-H",
         "llm_backbone": "Yi-34B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 68.8, "year": 2024, "source": "young2024yi"},
        {"model": "Yi-VL-34B", "params_b": 34.0, "vision_encoder": "CLIP-ViT-H",
         "llm_backbone": "Yi-34B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 43.5, "year": 2024, "source": "young2024yi"},
        {"model": "Yi-VL-34B", "params_b": 34.0, "vision_encoder": "CLIP-ViT-H",
         "llm_backbone": "Yi-34B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 2050.2, "year": 2024, "source": "young2024yi"},

        # ====================================================================
        # Fuyu-8B (Adept, 2023)
        # ====================================================================
        {"model": "Fuyu-8B", "params_b": 8.0, "vision_encoder": "linear_projection",
         "llm_backbone": "Persimmon-8B", "training_strategy": "pretraining_alignment",
         "benchmark": "MMBench", "score": 10.7, "year": 2023, "source": "bavishi2023fuyu"},
        {"model": "Fuyu-8B", "params_b": 8.0, "vision_encoder": "linear_projection",
         "llm_backbone": "Persimmon-8B", "training_strategy": "pretraining_alignment",
         "benchmark": "MM-Vet", "score": 21.4, "year": 2023, "source": "bavishi2023fuyu"},
        {"model": "Fuyu-8B", "params_b": 8.0, "vision_encoder": "linear_projection",
         "llm_backbone": "Persimmon-8B", "training_strategy": "pretraining_alignment",
         "benchmark": "MME", "score": 728.6, "year": 2023, "source": "bavishi2023fuyu"},

        # ====================================================================
        # Monkey (Li et al., 2024)
        # ====================================================================
        {"model": "Monkey", "params_b": 9.8, "vision_encoder": "ViT-BigHuge",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 72.4, "year": 2024, "source": "li2024monkey"},
        {"model": "Monkey", "params_b": 9.8, "vision_encoder": "ViT-BigHuge",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 64.3, "year": 2024, "source": "li2024monkey"},
        {"model": "Monkey", "params_b": 9.8, "vision_encoder": "ViT-BigHuge",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 33.0, "year": 2024, "source": "li2024monkey"},
        {"model": "Monkey", "params_b": 9.8, "vision_encoder": "ViT-BigHuge",
         "llm_backbone": "Qwen-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 67.6, "year": 2024, "source": "li2024monkey"},

        # ====================================================================
        # LLaVA-NeXT-7B (Liu et al., 2024)
        # ====================================================================
        {"model": "LLaVA-NeXT-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 67.4, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 70.2, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 43.9, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1588.7, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 64.9, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-7B", "params_b": 7.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-7B", "training_strategy": "instruction_tuning",
         "benchmark": "POPE", "score": 86.5, "year": 2024, "source": "liu2024llavanext"},

        # ====================================================================
        # LLaVA-NeXT-13B (Liu et al., 2024)
        # ====================================================================
        {"model": "LLaVA-NeXT-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 70.0, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 71.9, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 48.4, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1608.2, "year": 2024, "source": "liu2024llavanext"},
        {"model": "LLaVA-NeXT-13B", "params_b": 13.0, "vision_encoder": "CLIP-ViT-L",
         "llm_backbone": "Vicuna-13B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 67.1, "year": 2024, "source": "liu2024llavanext"},

        # ====================================================================
        # Cambrian-1-8B (Tong et al., 2024)
        # ====================================================================
        {"model": "Cambrian-1-8B", "params_b": 8.0, "vision_encoder": "multi_encoder",
         "llm_backbone": "LLaMA-3-8B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 75.9, "year": 2024, "source": "tong2024cambrian"},
        {"model": "Cambrian-1-8B", "params_b": 8.0, "vision_encoder": "multi_encoder",
         "llm_backbone": "LLaMA-3-8B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 73.0, "year": 2024, "source": "tong2024cambrian"},
        {"model": "Cambrian-1-8B", "params_b": 8.0, "vision_encoder": "multi_encoder",
         "llm_backbone": "LLaMA-3-8B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 49.0, "year": 2024, "source": "tong2024cambrian"},
        {"model": "Cambrian-1-8B", "params_b": 8.0, "vision_encoder": "multi_encoder",
         "llm_backbone": "LLaMA-3-8B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1845.3, "year": 2024, "source": "tong2024cambrian"},

        # ====================================================================
        # DeepSeek-VL-7B (Lu et al., 2024)
        # ====================================================================
        {"model": "DeepSeek-VL-7B", "params_b": 7.3, "vision_encoder": "SigLIP-L",
         "llm_backbone": "DeepSeek-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MMBench", "score": 73.2, "year": 2024, "source": "lu2024deepseekvl"},
        {"model": "DeepSeek-VL-7B", "params_b": 7.3, "vision_encoder": "SigLIP-L",
         "llm_backbone": "DeepSeek-7B", "training_strategy": "instruction_tuning",
         "benchmark": "SEED-Bench", "score": 70.4, "year": 2024, "source": "lu2024deepseekvl"},
        {"model": "DeepSeek-VL-7B", "params_b": 7.3, "vision_encoder": "SigLIP-L",
         "llm_backbone": "DeepSeek-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MM-Vet", "score": 41.5, "year": 2024, "source": "lu2024deepseekvl"},
        {"model": "DeepSeek-VL-7B", "params_b": 7.3, "vision_encoder": "SigLIP-L",
         "llm_backbone": "DeepSeek-7B", "training_strategy": "instruction_tuning",
         "benchmark": "MME", "score": 1765.4, "year": 2024, "source": "lu2024deepseekvl"},
        {"model": "DeepSeek-VL-7B", "params_b": 7.3, "vision_encoder": "SigLIP-L",
         "llm_backbone": "DeepSeek-7B", "training_strategy": "instruction_tuning",
         "benchmark": "TextVQA", "score": 64.7, "year": 2024, "source": "lu2024deepseekvl"},
    ]
    
    records = records + _load_new_models()
    df = pd.DataFrame(records)

    # Add derived columns
    df["log_params"] = np.log10(df["params_b"])
    df["scale_category"] = pd.cut(
        df["params_b"],
        bins=[0, 10, 50, 100, 2000],
        labels=["Small (<10B)", "Medium (10-50B)", "Large (50-100B)", "Very Large (>100B)"]
    )
    
    # Map vision encoder to simplified categories
    encoder_map = {
        "CLIP-ViT-L": "CLIP-family",
        "CLIP-ViT-H": "CLIP-family",
        "EVA-ViT-G": "EVA-family",
        "EVA2-CLIP-E": "EVA-family",
        "ViT-bigG": "ViT-large",
        "ViT-L": "ViT-large",
        "ViT-BigHuge": "ViT-large",
        "InternViT-6B": "InternViT",
        "SigLIP-L": "SigLIP",
        "linear_projection": "linear_projection",
        "multi_encoder": "multi_encoder",
        "proprietary": "proprietary",
    }
    df["encoder_family"] = df["vision_encoder"].map(encoder_map).fillna(df["vision_encoder"])

    # Attach provenance fields so every record carries its own source webpage,
    # venue and peer-review status (exported in Table 13).
    _blank = ("", "", "", "")
    df["source_url"] = df["source"].map(lambda s: SOURCE_REGISTRY.get(s, _blank)[0])
    df["source_venue"] = df["source"].map(lambda s: SOURCE_REGISTRY.get(s, _blank)[1])
    df["peer_reviewed"] = df["source"].map(lambda s: SOURCE_REGISTRY.get(s, _blank)[2])
    df["manuscript_ref"] = df["source"].map(lambda s: SOURCE_REGISTRY.get(s, _blank)[3])

    return df


def get_benchmark_metadata() -> dict:
    """Return metadata about each benchmark (scale, task type, etc.)."""
    return {
        "MMBench": {
            "full_name": "MMBench",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "comprehensive",
            "description": "Multi-ability benchmark covering perception, reasoning, and knowledge"
        },
        "SEED-Bench": {
            "full_name": "SEED-Bench",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "comprehensive",
            "description": "Spatial and temporal understanding in image and video"
        },
        "MM-Vet": {
            "full_name": "MM-Vet",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "open_ended",
            "description": "Open-ended visual chat evaluation using GPT-4 scoring"
        },
        "MME": {
            "full_name": "MME",
            "scale": "raw_score",
            "max_score": 2800.0,
            "task_type": "comprehensive",
            "description": "Perception and cognition abilities via yes/no questions"
        },
        "TextVQA": {
            "full_name": "TextVQA",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "ocr",
            "description": "Visual question answering requiring text reading in images"
        },
        "POPE": {
            "full_name": "POPE",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "hallucination",
            "description": "Polling-based object probing for hallucination evaluation"
        },
        "VQAv2": {
            "full_name": "VQAv2",
            "scale": "percentage",
            "max_score": 100.0,
            "task_type": "vqa",
            "description": "General visual question answering on natural images"
        },
    }


if __name__ == "__main__":
    df = get_benchmark_data()
    print(f"Total records: {len(df)}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique benchmarks: {df['benchmark'].nunique()}")
    print(f"\nModels: {sorted(df['model'].unique())}")
    print(f"\nBenchmarks: {sorted(df['benchmark'].unique())}")
    print(f"\nRecords per benchmark:")
    print(df.groupby("benchmark").size().sort_values(ascending=False))
