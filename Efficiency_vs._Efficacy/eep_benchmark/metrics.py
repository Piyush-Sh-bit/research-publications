import Levenshtein

def normalized_levenshtein_similarity(pred: str, gt: str) -> float:
    """
    Computes Normalized Levenshtein Similarity (NLS).
    NLS = 1 - (EditDistance(pred, gt) / max(len(pred), len(gt)))
    """
    pred = str(pred).strip().lower()
    gt = str(gt).strip().lower()
    
    if len(pred) == 0 and len(gt) == 0:
        return 1.0
        
    max_len = max(len(pred), len(gt))
    dist = Levenshtein.distance(pred, gt)
    
    return 1.0 - (dist / max_len)

def anls_score(pred: str, gt: str, threshold: float = 0.5) -> float:
    """
    Average Normalized Levenshtein Similarity (ANLS).
    Defined as NLS if NLS >= threshold, else 0.0.
    """
    nls = normalized_levenshtein_similarity(pred, gt)
    return nls if nls >= threshold else 0.0

def compute_eep(eta: float, gamma: float, omega: float, l_ms: float, psi: float = 1.15) -> float:
    """
    Computes the Efficiency-Efficacy Pareto (EEP) Score.
    S_EEP = (eta * ln(1 + gamma / omega)) / ((1 + l_s) * psi)
    
    Args:
        eta: Efficacy (e.g., mean ANLS)
        gamma: Throughput in TPS (tokens/s or docs/s)
        omega: Peak VRAM in GB
        l_ms: End-to-end generation latency in milliseconds
        psi: Complexity-scaling factor
    """
    import math
    l_s = l_ms / 1000.0
    if omega == 0:
        return 0.0
    return (eta * math.log(1 + (gamma / omega))) / ((1 + l_s) * psi)
