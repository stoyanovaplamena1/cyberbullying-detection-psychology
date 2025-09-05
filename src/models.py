# src/models.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import time
import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score
)


from paths import RANDOM_STATE

# -----------------------------------------------------------------------------
# CB binary: calibrated elastic-net logistic (train-only CV)
# -----------------------------------------------------------------------------
def fit_calibrated_logreg_binary(
    Xtr, ytr,
    cv: int = 3,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-3,
    random_state: int = RANDOM_STATE,
) -> CalibratedClassifierCV:
    """
    Train elastic-net logistic (SAGA) wrapped with sigmoid calibration (internal CV on TRAIN).
    Returns a CalibratedClassifierCV with predict_proba available.
    """
    base = LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=l1_ratio, C=C, class_weight="balanced",
        max_iter=max_iter, tol=tol, n_jobs=-1, random_state=random_state
    )
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)
    clf.fit(Xtr, ytr)
    return clf

def predict_proba_binary(model, X) -> np.ndarray:
    """Return positive-class probabilities (shape (n,))."""
    return model.predict_proba(X)[:, 1]

# -----------------------------------------------------------------------------
# TX multilabel: fast OvR with SGD + prefit calibration (sigmoid or isotonic)
# -----------------------------------------------------------------------------
def fit_ovr_sgd_calibrated(
    Xtr, Ytr: np.ndarray,
    calib_method: str = "sigmoid",     # or "isotonic"
    calib_frac: float = 0.10,
    random_state: int = RANDOM_STATE,
    sgd_alpha: float = 1e-4,
    sgd_l1_ratio: float = 0.15,
    sgd_max_iter: int = 20,
) -> List[CalibratedClassifierCV]:
    """
    Fit one calibrated binary classifier per column of Ytr using SGD(log_loss) base.
    Calibration is 'prefit' on a small slice of TRAIN (no dev leakage).
    Returns list of calibrated estimators in label order.
    """
    L = Ytr.shape[1]
    models: List[CalibratedClassifierCV] = []

    for j in range(L):
        yj = Ytr[:, j].astype(int)
        # robust stratification for tiny positives
        counts = np.bincount(yj, minlength=2)
        strat = yj if (counts.min() >= 2 and counts.min() * calib_frac >= 1) else None

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            Xtr, yj, test_size=calib_frac, stratify=strat, random_state=random_state
        )

        base = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            alpha=sgd_alpha, l1_ratio=sgd_l1_ratio,
            class_weight="balanced",
            max_iter=sgd_max_iter, tol=1e-3,
            early_stopping=True, n_iter_no_change=3,
            validation_fraction=0.1, average=True,
            random_state=random_state
        )
        base.fit(X_fit, y_fit)

        # Handle sklearn>=1.6 deprecation of cv="prefit"
        try:
            from sklearn.calibration import FrozenEstimator  # type: ignore
            cal = CalibratedClassifierCV(FrozenEstimator(base), method=calib_method)
        except Exception:
            cal = CalibratedClassifierCV(base, method=calib_method, cv="prefit")

        cal.fit(X_cal, y_cal)
        models.append(cal)

    return models

def predict_proba_multilabel(models: List[CalibratedClassifierCV], X) -> np.ndarray:
    """Return probability matrix of shape (n_samples, n_labels)."""
    cols = [m.predict_proba(X)[:, 1] for m in models]
    return np.column_stack(cols)

# -----------------------------------------------------------------------------
# CB multiclass options: fast LinearSVC OR lighter OvR logistic
# -----------------------------------------------------------------------------
def fit_multiclass_linear_svc(
    Xtr, ytr,
    C: float = 1.0,
    class_weight: str | dict = "balanced",
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> LinearSVC:
    """
    Fast linear SVM head for multiclass (no probabilities, strong macro-F1 baseline).
    """
    clf = LinearSVC(C=C, class_weight=class_weight, max_iter=max_iter)
    clf.fit(Xtr, ytr)
    return clf

def fit_multiclass_logreg_ovr(
    Xtr, ytr,
    C: float = 1.0,
    class_weight: str | dict = "balanced",
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
):
    """
    L2 OvR logistic (liblinear) — lighter than multinomial SAGA, provides predict_proba if needed.
    """
    clf = LogisticRegression(
        solver="liblinear", multi_class="ovr",
        penalty="l2", C=C, class_weight=class_weight,
        max_iter=max_iter, n_jobs=-1, random_state=random_state
    )
    clf.fit(Xtr, ytr)
    return clf

# -----------------------------------------------------------------------------
# Cross-domain helper: evaluate zero-shot transfer for a binary task
# -----------------------------------------------------------------------------
def eval_zero_shot_binary(
    model, X_target, y_target: np.ndarray, threshold: float
) -> Dict[str, float]:
    """
    Score a calibrated binary model on a target domain using a frozen threshold.
    Returns PR-AUC, ROC-AUC, precision, recall, F1.
    """
    p = predict_proba_binary(model, X_target)
    pr_auc = float(average_precision_score(y_target, p))
    roc_auc = float(roc_auc_score(y_target, p))
    yhat = (p >= threshold).astype(int)
    prec = float(precision_score(y_target, yhat, zero_division=0))
    rec  = float(recall_score(y_target, yhat, zero_division=0))
    f1   = float(f1_score(y_target, yhat, zero_division=0))
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "precision": prec, "recall": rec, "f1": f1}
