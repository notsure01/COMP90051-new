"""
All evaluation metrics implemented from scratch using only NumPy.
No scikit-learn or other third-party metric libraries are used.

Implemented:
  confusion_matrix   – C×C count matrix
  accuracy           – fraction of correct predictions
  precision          – macro-averaged or per-class
  recall             – macro-averaged or per-class
  f1_score           – macro-averaged or per-class
  summarise_metrics  – convenience wrapper returning a flat dict
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Core building block
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: list[int] | np.ndarray,
                     y_pred: list[int] | np.ndarray,
                     num_classes: int) -> np.ndarray:
    """
    Build a C×C confusion matrix without using any library routine.

    Element [i, j] is the number of samples with true label i predicted as j.

    Parameters
    ----------
    y_true      : ground-truth integer labels, shape (N,)
    y_pred      : predicted integer labels,    shape (N,)
    num_classes : C, total number of classes

    Returns
    -------
    cm : np.ndarray of shape (C, C), dtype int64
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Per-class metrics derived from the confusion matrix
# ---------------------------------------------------------------------------

def _per_class_precision(cm: np.ndarray) -> np.ndarray:
    """
    Precision for each class: TP / (TP + FP).
    Returns 0 for a class with no predicted positives (avoid division by 0).
    """
    num_classes = cm.shape[0]
    prec = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        denom = tp + fp
        prec[c] = tp / denom if denom > 0 else 0.0
    return prec


def _per_class_recall(cm: np.ndarray) -> np.ndarray:
    """
    Recall for each class: TP / (TP + FN).
    Returns 0 for a class with no actual positives.
    """
    num_classes = cm.shape[0]
    rec = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        denom = tp + fn
        rec[c] = tp / denom if denom > 0 else 0.0
    return rec


def _per_class_f1(prec: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """F1 = 2 * P * R / (P + R).  Returns 0 when both P and R are 0."""
    num_classes = len(prec)
    f1 = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        denom = prec[c] + rec[c]
        f1[c] = 2.0 * prec[c] * rec[c] / denom if denom > 0 else 0.0
    return f1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def accuracy(y_true: list[int] | np.ndarray,
             y_pred: list[int] | np.ndarray) -> float:
    """Overall accuracy: fraction of exactly-correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true: list[int] | np.ndarray,
              y_pred: list[int] | np.ndarray,
              num_classes: int,
              average: str = "macro") -> float | np.ndarray:
    """
    Precision score.

    Parameters
    ----------
    average : "macro" returns the unweighted mean across classes;
              "none"  returns a per-class array of length C.
    """
    cm   = confusion_matrix(y_true, y_pred, num_classes)
    prec = _per_class_precision(cm)
    return float(prec.mean()) if average == "macro" else prec


def recall(y_true: list[int] | np.ndarray,
           y_pred: list[int] | np.ndarray,
           num_classes: int,
           average: str = "macro") -> float | np.ndarray:
    """
    Recall score.

    Parameters
    ----------
    average : "macro" or "none" (see `precision`).
    """
    cm  = confusion_matrix(y_true, y_pred, num_classes)
    rec = _per_class_recall(cm)
    return float(rec.mean()) if average == "macro" else rec


def f1_score(y_true: list[int] | np.ndarray,
             y_pred: list[int] | np.ndarray,
             num_classes: int,
             average: str = "macro") -> float | np.ndarray:
    """
    F1 score.

    Parameters
    ----------
    average : "macro" or "none" (see `precision`).
    """
    cm   = confusion_matrix(y_true, y_pred, num_classes)
    prec = _per_class_precision(cm)
    rec  = _per_class_recall(cm)
    f1   = _per_class_f1(prec, rec)
    return float(f1.mean()) if average == "macro" else f1


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def summarise_metrics(y_true: list[int] | np.ndarray,
                      y_pred: list[int] | np.ndarray,
                      num_classes: int) -> dict[str, float]:
    """
    Compute all four scalar metrics in a single pass over the confusion matrix.

    Returns
    -------
    dict with keys: "accuracy", "precision", "recall", "f1"
    """
    cm   = confusion_matrix(y_true, y_pred, num_classes)
    prec = _per_class_precision(cm)
    rec  = _per_class_recall(cm)
    f1   = _per_class_f1(prec, rec)
    acc  = float(np.trace(cm)) / float(cm.sum()) if cm.sum() > 0 else 0.0

    return {
        "accuracy":  acc,
        "precision": float(prec.mean()),
        "recall":    float(rec.mean()),
        "f1":        float(f1.mean()),
    }


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]
                           ) -> dict[str, dict[str, float]]:
    """
    Given a list of per-fold metric dicts (from nested CV outer folds),
    compute mean and standard deviation for each metric.

    Returns
    -------
    dict[metric_name] → {"mean": float, "std": float}
    """
    keys = list(fold_metrics[0].keys())
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = np.array([fm[key] for fm in fold_metrics])
        result[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return result
