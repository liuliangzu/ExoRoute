"""
Evaluation metrics for REGATE experiments.

Implements AUC, AP, ECE, Precision@K as described in the paper.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Optional


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_score)


def compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Average Precision."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_score)


def compute_precision_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k_percent: float = 5.0
) -> float:
    """
    Compute Precision@K% budget.
    Select top K% by predicted score and compute precision.
    """
    n = len(y_true)
    k = max(1, int(n * k_percent / 100.0))
    top_k_idx = np.argsort(y_score)[-k:]
    return y_true[top_k_idx].mean()


def compute_ece(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width bins.
    
    ECE = sum_{m=1}^{M} (|B_m| / n) * |acc(B_m) - conf(B_m)|
    
    As described in Appendix A.2 of the paper.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for m in range(n_bins):
        lower = bin_boundaries[m]
        upper = bin_boundaries[m + 1]

        # Find samples in this bin
        if m < n_bins - 1:
            in_bin = (y_score > lower) & (y_score <= upper)
        else:
            in_bin = (y_score > lower) & (y_score <= upper + 1e-8)

        bin_size = in_bin.sum()
        if bin_size == 0:
            continue

        bin_acc = y_true[in_bin].mean()
        bin_conf = y_score[in_bin].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return ece


def evaluate_all(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_percents: list = [1.0, 5.0],
) -> Dict[str, float]:
    """
    Compute all metrics.
    
    Returns:
        dict with keys: auc, ap, ece, prec@1%, prec@5%
    """
    results = {
        "auc": compute_auc(y_true, y_score),
        "ap": compute_ap(y_true, y_score),
        "ece": compute_ece(y_true, y_score),
    }
    for k in k_percents:
        results[f"prec@{k}%"] = compute_precision_at_k(y_true, y_score, k)
    return results
