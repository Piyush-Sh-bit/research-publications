"""Hidden-state extraction. Frozen models only -- pure inference.

Returns, for a list of sentences, a list of pooled representation matrices
(one (n, hidden_dim) tensor per layer, including the embedding layer 0).
"""
from __future__ import annotations

import random
from typing import List

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_model(name: str, load_in_4bit: bool = False, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Build the config first and GUARANTEE pad_token_id exists before loading:
    # some configs (e.g. Phi) omit it and raise AttributeError inside
    # from_pretrained. Setting it on the config here fixes that at the source.
    config = AutoConfig.from_pretrained(name)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    config.output_hidden_states = True
    kwargs = dict(config=config)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(name, **kwargs)
    if not load_in_4bit:
        dev = device if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
    model.eval()
    return tok, model


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over the token axis. hidden: (B, T, H)."""
    m = mask.unsqueeze(-1).float()
    summed = (hidden * m).sum(1)
    counts = m.sum(1).clamp(min=1e-6)
    return summed / counts


def token_dropout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
    p: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop a fraction p of non-pad tokens (replace by pad, mask off)."""
    g = torch.Generator().manual_seed(seed)
    ids = input_ids.clone()
    mask = attention_mask.clone()
    drop = (torch.rand(ids.shape, generator=g) < p) & (mask.bool())
    ids[drop] = pad_id
    mask[drop] = 0
    return ids, mask


@torch.no_grad()
def extract_all_layers(
    sentences: List[str],
    tok,
    model,
    max_length: int = 64,
    batch_size: int = 16,
    perturb: bool = False,
    token_dropout_p: float = 0.15,
    seed: int = 0,
) -> List[torch.Tensor]:
    """Return list[ tensor(n, H) ] indexed by layer (0 = embeddings)."""
    device = next(model.parameters()).device
    per_layer: List[List[torch.Tensor]] = None
    pad_id = tok.pad_token_id

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if perturb:
            ids, mask = token_dropout(
                ids, mask, pad_id, token_dropout_p, seed + start
            )
        ids, mask = ids.to(device), mask.to(device)
        out = model(input_ids=ids, attention_mask=mask)
        hs = out.hidden_states  # tuple(L+1) of (B, T, H)

        if per_layer is None:
            per_layer = [[] for _ in range(len(hs))]
        for li, h in enumerate(hs):
            pooled = _mean_pool(h.float(), mask).cpu()
            per_layer[li].append(pooled)

    return [torch.cat(chunks, dim=0) for chunks in per_layer]


@torch.no_grad()
def curvature_per_layer(
    sentences: List[str],
    tok,
    model,
    max_length: int = 64,
    batch_size: int = 16,
    max_sents: int = 256,
) -> List[float]:
    """Mean representation-trajectory curvature per layer (Skean et al., 2025).

    Along each token sequence, take difference vectors d_i = h_{i+1} - h_i, then
    the curvature at position i is the angle between consecutive differences,
    acos(cos(d_{i-1}, d_i)). We average over valid (non-pad) positions and over
    sentences. This is token-level (not pooled) and non-monotone in depth -- a
    genuinely different signal from the pooled geometry / intrinsic dimension.
    """
    device = next(model.parameters()).device
    sents = sentences[:max_sents]
    ssum = None
    scnt = None
    for start in range(0, len(sents), batch_size):
        batch = sents[start:start + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        hs = model(input_ids=ids, attention_mask=mask).hidden_states  # (L+1) x (B,T,H)
        if ssum is None:
            ssum = [0.0] * len(hs); scnt = [0.0] * len(hs)
        m = mask.float()
        valid = (m[:, :-2] * m[:, 1:-1] * m[:, 2:]) if m.shape[1] >= 3 else None
        for li, h in enumerate(hs):
            if valid is None:
                continue
            h = h.float()
            d = h[:, 1:, :] - h[:, :-1, :]                       # (B,T-1,H)
            cos = torch.nn.functional.cosine_similarity(
                d[:, :-1, :], d[:, 1:, :], dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
            ang = torch.acos(cos)                                # (B,T-2)
            ssum[li] += float((ang * valid).sum().item())
            scnt[li] += float(valid.sum().item())
    return [ssum[li] / max(scnt[li], 1e-9) for li in range(len(ssum))]
