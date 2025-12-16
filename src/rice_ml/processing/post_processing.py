from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike1D = Union[np.ndarray, Sequence]

__all__ = [
    # classification
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "roc_auc_score",
    "log_loss",
    # regression
    "mse",
    "rmse",
    "mae",
    "r2_score",
]


def _as_1d(a: ArrayLike1D, *, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _validate_equal_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")


def _validate_numeric(a: np.ndarray, *, name: str) -> np.ndarray:
    if not np.issubdtype(a.dtype, np.number):
        raise TypeError(f"{name} must contain only numeric values.")
    return a.astype(float, copy=False)


# ---------------------------------------------------------------------------
# Classification: basic metrics
# ---------------------------------------------------------------------------


def accuracy_score(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    y_true = _as_1d(y_true, name="y_true")
    y_pred = _as_1d(y_pred, name="y_pred")
    _validate_equal_length(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: ArrayLike1D,
    y_pred: ArrayLike1D,
    *,
    labels: Optional[Sequence] = None,
) -> np.ndarray:
    y_true = _as_1d(y_true, name="y_true")
    y_pred = _as_1d(y_pred, name="y_pred")
    _validate_equal_length(y_true, y_pred)

    if labels is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.asarray(list(labels))

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)

    # mapping for quick lookup
    index = {c: i for i, c in enumerate(classes.tolist())}
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t not in index or p not in index:
            # ignore samples containing unknown labels
            continue
        cm[index[t], index[p]] += 1
    return cm


def _tp_fp_fn(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.zeros(len(labels), dtype=int)
    fp = np.zeros(len(labels), dtype=int)
    fn = np.zeros(len(labels), dtype=int)
    for i, c in enumerate(labels.tolist()):
        tp[i] = int(np.sum((y_true == c) & (y_pred == c)))
        fp[i] = int(np.sum((y_true != c) & (y_pred == c)))
        fn[i] = int(np.sum((y_true == c) & (y_pred != c)))
    return tp, fp, fn


def precision_score(y_true: ArrayLike1D, y_pred: ArrayLike1D, *, average: str = "binary") -> float:
    y_true = _as_1d(y_true, name="y_true")
    y_pred = _as_1d(y_pred, name="y_pred")
    _validate_equal_length(y_true, y_pred)

    average = str(average)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        if len(labels) != 2:
            raise ValueError("average='binary' is only defined for 2 classes.")
        pos = 1
        tp = np.sum((y_true == pos) & (y_pred == pos))
        fp = np.sum((y_true != pos) & (y_pred == pos))
        denom = tp + fp
        return float(tp / denom) if denom != 0 else 0.0

    tp, fp, fn = _tp_fp_fn(y_true, y_pred, labels)
    if average == "macro":
        per = np.where(tp + fp == 0, 0.0, tp / (tp + fp))
        return float(per.mean())
    if average == "micro":
        tp_all = tp.sum()
        fp_all = fp.sum()
        denom = tp_all + fp_all
        return float(tp_all / denom) if denom != 0 else 0.0
    raise ValueError("average must be one of {'binary','macro','micro'}.")


def recall_score(y_true: ArrayLike1D, y_pred: ArrayLike1D, *, average: str = "binary") -> float:
    y_true = _as_1d(y_true, name="y_true")
    y_pred = _as_1d(y_pred, name="y_pred")
    _validate_equal_length(y_true, y_pred)

    average = str(average)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        if len(labels) != 2:
            raise ValueError("average='binary' is only defined for 2 classes.")
        pos = 1
        tp = np.sum((y_true == pos) & (y_pred == pos))
        fn = np.sum((y_true == pos) & (y_pred != pos))
        denom = tp + fn
        return float(tp / denom) if denom != 0 else 0.0

    tp, fp, fn = _tp_fp_fn(y_true, y_pred, labels)
    if average == "macro":
        per = np.where(tp + fn == 0, 0.0, tp / (tp + fn))
        return float(per.mean())
    if average == "micro":
        tp_all = tp.sum()
        fn_all = fn.sum()
        denom = tp_all + fn_all
        return float(tp_all / denom) if denom != 0 else 0.0
    raise ValueError("average must be one of {'binary','macro','micro'}.")


def f1_score(y_true: ArrayLike1D, y_pred: ArrayLike1D, *, average: str = "binary") -> float:
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    return float(2 * p * r / (p + r)) if (p + r) != 0 else 0.0


# ---------------------------------------------------------------------------
# Classification: ROC AUC + log loss
# ---------------------------------------------------------------------------


def roc_auc_score(y_true: ArrayLike1D, y_score: ArrayLike1D) -> float:
    """Binary ROC AUC using the rank statistic (Mann–Whitney U)."""
    y = _as_1d(y_true, name="y_true")
    s = _as_1d(y_score, name="y_score")
    _validate_equal_length(y, s)
    s = _validate_numeric(np.asarray(s), name="y_score")

    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError("roc_auc_score is only defined for binary y_true.")

    # treat label '1' as positive class
    y01 = (y == 1).astype(int)
    n_pos = int(y01.sum())
    n_neg = int(len(y01) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc_score is undefined with only one class present.")

    # ranks with average for ties
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    sum_pos_ranks = ranks[y01 == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def log_loss(y_true: ArrayLike1D, y_prob: Union[ArrayLike1D, np.ndarray], *, eps: float = 1e-15) -> float:
    """
    Log loss for binary (1D prob of class 1) or multiclass (2D prob matrix).

    For 2D probabilities, rows must sum to ~1 (no auto-renormalization).
    """
    y = _as_1d(y_true, name="y_true").astype(int, copy=False)
    p = np.asarray(y_prob)
    if p.ndim == 1:
        p1 = _validate_numeric(_as_1d(p, name="y_prob"), name="y_prob")
        _validate_equal_length(y, p1)
        if np.any((p1 < 0) | (p1 > 1)):
            raise ValueError("Probabilities must be within [0, 1].")
        # allow exact 1.0 so perfect predictions yield exact 0.0 loss
        p1 = np.clip(p1, eps, 1.0)
        # probability of the true class
        pt = np.where(y == 1, p1, 1 - p1)
        if np.any((y != 0) & (y != 1)):
            raise ValueError("Binary log_loss requires y_true in {0,1}.")
        pt = np.clip(pt, eps, 1.0)
        return float(-np.mean(np.log(pt)))

    if p.ndim != 2:
        raise ValueError("y_prob must be 1D (binary) or 2D (multiclass).")
    if p.shape[0] != len(y):
        raise ValueError("y_true and y_prob must have the same number of samples.")
    p = _validate_numeric(p, name="y_prob")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be within [0, 1].")
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Each probability row must sum to 1.")

    n_classes = p.shape[1]
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError("y_true contains invalid class indices.")
    # allow exact 1.0 so one-hot probabilities yield exact 0.0 loss
    p = np.clip(p, eps, 1.0)
    pt = p[np.arange(len(y)), y]
    return float(-np.mean(np.log(pt)))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def mse(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    y_true = _validate_numeric(_as_1d(y_true, name="y_true"), name="y_true")
    y_pred = _validate_numeric(_as_1d(y_pred, name="y_pred"), name="y_pred")
    _validate_equal_length(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    y_true = _validate_numeric(_as_1d(y_true, name="y_true"), name="y_true")
    y_pred = _validate_numeric(_as_1d(y_pred, name="y_pred"), name="y_pred")
    _validate_equal_length(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    y_true = _validate_numeric(_as_1d(y_true, name="y_true"), name="y_true")
    y_pred = _validate_numeric(_as_1d(y_pred, name="y_pred"), name="y_pred")
    _validate_equal_length(y_true, y_pred)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0.0:
        if ss_res == 0.0:
            return 1.0
        raise ValueError("R^2 is undefined when y_true is constant.")
    return float(1 - ss_res / ss_tot)
