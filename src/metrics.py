# src/metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple, Callable
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)

# -----------------------------
# Threshold selection (binary)
# -----------------------------
def tune_f1(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Sweep the PR curve on (y_true, scores) and return:
      (best_threshold_for_F1, best_F1_value).
    """
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    best = int(np.nanargmax(f1s[:-1]))  # ignore sentinel
    return float(thr[best]), float(f1s[best])

def tune_precision_floor(y_true: np.ndarray, scores: np.ndarray, target_p: float = 0.95) -> float:
    """
    Choose the smallest threshold achieving precision >= target_p on dev.
    If unattainable, fall back to the highest-precision point.
    """
    prec, rec, thr = precision_recall_curve(y_true, scores)
    mask = prec[:-1] >= target_p
    if np.any(mask):
        idxs = np.where(mask)[0]
        best = idxs[np.argmax(rec[:-1][idxs])]
    else:
        best = int(np.argmax(prec[:-1]))
    return float(thr[best])

def p_at_frac(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.10) -> Tuple[float, int]:
    """
    Precision@k where k = ceil(frac * n). Returns (precision_at_k, k).
    """
    n = len(y_true)
    k = max(1, int(frac * n))
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].mean()), k

# -----------------------------
# Binary evaluation helpers
# -----------------------------
def eval_binary(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Apply threshold and compute PR-AUC, ROC-AUC, F1, precision, recall.
    """
    yhat = (scores >= thr).astype(int)
    return dict(
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        f1=float(f1_score(y_true, yhat, zero_division=0)),
        precision=float(precision_score(y_true, yhat, zero_division=0)),
        recall=float(recall_score(y_true, yhat, zero_division=0)),
    )

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return (tp, fp, fn, tn).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int(tp), int(fp), int(fn), int(tn)

# -----------------------------
# Bootstrap uncertainty (CI)
# -----------------------------
def bootstrap_ci_binary(
    y_true: np.ndarray,
    scores: np.ndarray,
    thr: float,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap mean and 95% CI for a thresholded binary metric.
    metric_fn takes (y_true, y_pred) and returns a scalar (e.g., f1_score with zero_division=0).
    Returns (mean, lo, hi).
    """
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, sb = y_true[idx], scores[idx]
        yhat = (sb >= thr).astype(int)
        vals.append(float(metric_fn(yb, yhat)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)

def bootstrap_ci_macro_f1(
    Y_true: np.ndarray,                      # (n, L)
    P_scores: np.ndarray,                    # (n, L) probabilities
    thresholds: Dict[str, float],            # per-label threshold map
    label_order: List[str],
    n_boot: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap mean and 95% CI for macro-F1 in a multilabel setting using fixed per-label thresholds.
    """
    rng = np.random.RandomState(seed)
    n = Y_true.shape[0]
    L = len(label_order)
    vals = []
    thr_vec = np.array([thresholds[label_order[j]] for j in range(L)], dtype=float)

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        Yb = Y_true[idx]
        Pb = P_scores[idx]
        Yhat = (Pb >= thr_vec).astype(int)
        vals.append(float(f1_score(Yb, Yhat, average="macro", zero_division=0)))

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)

# -----------------------------
# Multilabel evaluation (flat OvR)
# -----------------------------
def eval_multilabel_at_thresholds(
    Y_true: np.ndarray,                      # (n, L)
    P_scores: np.ndarray,                    # (n, L)
    thresholds: Dict[str, float],            # per-label thr
    label_order: List[str],
    include_pr_auc: bool = True,
) -> Dict[str, object]:
    """
    Compute micro/macro-F1 and per-label stats at given thresholds.
    Returns dict with 'micro_f1', 'macro_f1', and 'per_label' list of dicts.
    """
    L = len(label_order)
    thr_vec = np.array([thresholds[label_order[j]] for j in range(L)], dtype=float)
    Yhat = (P_scores >= thr_vec).astype(int)

    micro = float(f1_score(Y_true, Yhat, average="micro", zero_division=0))
    macro = float(f1_score(Y_true, Yhat, average="macro", zero_division=0))

    per_label = []
    for j, lab in enumerate(label_order):
        yt, yp = Y_true[:, j], Yhat[:, j]
        P = float(precision_score(yt, yp, zero_division=0))
        R = float(recall_score(yt, yp, zero_division=0))
        F = float(f1_score(yt, yp, zero_division=0))
        row = {"label": lab, "precision": P, "recall": R, "f1": F, "threshold": float(thr_vec[j])}
        if include_pr_auc:
            try:
                row["pr_auc"] = float(average_precision_score(yt, P_scores[:, j]))
            except Exception:
                row["pr_auc"] = float("nan")
        per_label.append(row)

    return {"micro_f1": micro, "macro_f1": macro, "per_label": per_label}

# -----------------------------
# Threshold selection (multilabel, per label)
# -----------------------------
def per_label_f1_thresholds(
    Y_dev: np.ndarray,            # (n_dev, L)
    P_dev: np.ndarray,            # (n_dev, L)
    label_order: List[str]
) -> Dict[str, float]:
    """
    For each label, choose the F1-opt threshold on dev.
    """
    thr = {}
    for j, lab in enumerate(label_order):
        t, _ = tune_f1(Y_dev[:, j], P_dev[:, j])
        thr[lab] = t
    return thr

def per_label_precision_floor_thresholds(
    Y_dev: np.ndarray,
    P_dev: np.ndarray,
    label_order: List[str],
    floors: Dict[str, float]
) -> Dict[str, float]:
    """
    For each label, choose the smallest threshold on dev achieving precision >= floors[label].
    """
    thr = {}
    for j, lab in enumerate(label_order):
        thr[lab] = tune_precision_floor(Y_dev[:, j], P_dev[:, j], target_p=float(floors[lab]))
    return thr
