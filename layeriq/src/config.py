"""Central configuration for LayerIQ experiments.

Everything here is chosen to run on a single Kaggle GPU (T4 16GB / P100 16GB).
Large models are loaded in 4-bit; only inference / probing is performed, so no
backbone training is ever required.
"""
from dataclasses import dataclass, field
from typing import List


# Models grouped by feasibility tier. Start with TIER_SMALL on free CPU/GPU,
# then move to TIER_MEDIUM (4-bit) once the pipeline is verified.
TIER_SMALL = [
    "google-bert/bert-base-uncased",      # 12L encoder
    "FacebookAI/roberta-base",            # 12L encoder
    "distilbert-base-uncased",            # 6L  encoder
    "google/electra-base-discriminator",  # 12L encoder (ELECTRA)
    "EleutherAI/pythia-410m",             # 24L decoder
    "EleutherAI/pythia-160m",             # 12L decoder
    "openai-community/gpt2",              # 12L decoder
    "facebook/opt-350m",                  # 24L decoder (OPT)
]

# Ungated, T4-friendly (<=2.8B, fp16), architecturally diverse, deeper than
# TIER_SMALL (24-32 layers) -- deeper nets sharpen the intrinsic-dimension /
# entropy structure the v2 criterion relies on.
TIER_MEDIUM = [
    "EleutherAI/pythia-1.4b",   # 24L GPT-NeoX
    "EleutherAI/pythia-2.8b",   # 32L GPT-NeoX
    "EleutherAI/gpt-neo-1.3B",  # 24L GPT-Neo
    "Qwen/Qwen2.5-1.5B",        # 28L Qwen2
    "Qwen/Qwen2.5-3B",          # 36L Qwen2
    "microsoft/phi-2",          # 32L Phi (2.7B)
    "facebook/opt-1.3b",        # 24L OPT
    "bigscience/bloom-1b7",     # 24L BLOOM
]
# NOTE: gated models (meta-llama/Llama-3.2-1B, mistralai/Mistral-7B-v0.3) need a
# Kaggle HF_TOKEN secret + accepted licence; 7B-class also needs load_in_4bit=True.
# Add them here and set load_in_4bit=True in the notebook to include them.

# Encoder EMBEDDING models. These are the setting where layer selection actually
# matters: the transfer-optimal layer is clearly NOT the last one (large
# last-layer gap), so a selector can demonstrably beat the last-layer default.
# All ungated, ~12 layers, T4-trivial.
TIER_ENCODERS = [
    "sentence-transformers/all-mpnet-base-v2",        # 12L MPNet
    "sentence-transformers/all-MiniLM-L12-v2",        # 12L MiniLM
    "sentence-transformers/all-MiniLM-L6-v2",         # 6L  MiniLM
    "sentence-transformers/paraphrase-mpnet-base-v2", # 12L MPNet
    "intfloat/e5-base-v2",                            # 12L BERT
    "intfloat/e5-small-v2",                           # 12L BERT
    "BAAI/bge-base-en-v1.5",                          # 12L BERT
    "thenlper/gte-base",                              # 12L BERT (GTE)
]

# Large (7B-class) models -- load in 4-bit to fit a T4 (set load_in_4bit=True in
# the notebook when running this tier). Ungated only; the low-ID / compression
# structure is expected to be sharpest at this scale.
TIER_LARGE = [
    "Qwen/Qwen2.5-7B",          # 28L Qwen2
    "EleutherAI/pythia-6.9b",   # 32L GPT-NeoX
    "EleutherAI/gpt-j-6b",      # 28L GPT-J
    "facebook/opt-6.7b",        # 32L OPT
]
# Gated large models (need HF_TOKEN + licence): meta-llama/Meta-Llama-3-8B,
# mistralai/Mistral-7B-v0.3. Add here once a token/secret is configured.


@dataclass
class ExperimentConfig:
    models: List[str] = field(default_factory=lambda: list(TIER_SMALL))

    # Unlabeled corpus used to compute LayerIQ (NO labels, NO training).
    unlabeled_corpus: str = "wikitext"
    unlabeled_subset: str = "wikitext-103-raw-v1"
    n_unlabeled: int = 512          # number of sentences for the criterion
    max_length: int = 64           # token cap (keeps memory small)

    # Downstream tasks used ONLY to validate the criterion's predictions.
    sts_task: str = "stsb"          # GLUE STS-B (regression, Spearman)
    # SentEval-style probing tasks; SUBJ is a classic where encoder mid-layers
    # clearly beat the last layer. A failed task is skipped (see run_experiment).
    # The loader also supports "mr", "cr", "sst5" if you want to add them.
    cls_tasks: tuple = ("sst2", "trec", "subj")

    # Downstream APPLICATION: paraphrase retrieval (semantic search). This is the
    # setting where layer choice pays off for decoder/LLM embedders. A single
    # label-free layer choice vs. the last-layer default, measured by accuracy@1.
    retrieval_task: str = "mrpc"    # mrpc | paws
    retrieval_limit: int = 500      # pool size (candidates per query)

    # Dataset sizes. Measured: the full 4-tier sweep runs in ~4h at these sizes,
    # well inside the Kaggle 12h GPU limit. Reported effect sizes are small
    # (~1 accuracy point), so the probes are run at full size -- shrinking them
    # only adds noise to the layer ranking.
    cls_n_train: int = 2000
    cls_n_test: int = 1000
    sts_limit: int = 1500           # full STS-B validation split

    # Perturbation for the neighborhood-stability signal.
    token_dropout: float = 0.15     # fraction of tokens masked/dropped
    perturb_seed: int = 0

    # k for nearest-neighbour self-retrieval stability.
    knn_k: int = 1

    load_in_4bit: bool = False      # set True for 7B-class models on Kaggle
    # Delete each model's weights from the HF cache once it has been measured.
    # The full sweep downloads ~90GB, which overruns the Kaggle disk; each model
    # is read once, so nothing is lost. Set False if you have disk to spare and
    # want to avoid re-downloading on a second run.
    purge_hf_cache: bool = True
    batch_size: int = 16
    device: str = "cuda"            # falls back to cpu automatically
    seed: int = 0
    out_dir: str = "results"


DEFAULT = ExperimentConfig()
