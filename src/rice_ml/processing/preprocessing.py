from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

__all__ = [
    "standardize",
    "minmax_scale",
    "maxabs_scale",
    "l2_normalize_rows",
    "train_test_split",
    "train_val_test_split",
]


def _as_2d_float(X: ArrayLike, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if arr.size == 0 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values.")
    return arr.astype(float, copy=False)


def _validate_random_state(random_state: Optional[int]) -> Optional[int]:
    if random_state is None:
        return None
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an int or None.")
    return int(random_state)


def standardize(
    X: ArrayLike,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Standardize features to zero mean and/or unit variance.

    Constant (zero-variance) features use a scale of 1.0.
    """
    Xf = _as_2d_float(X, name="X")

    mean = Xf.mean(axis=0) if with_mean else np.zeros(Xf.shape[1], dtype=float)
    if with_std:
        scale = Xf.std(axis=0)
        scale = np.where(scale == 0, 1.0, scale)
    else:
        scale = np.ones(Xf.shape[1], dtype=float)

    Z = (Xf - mean) / scale

    if return_params:
        params = {"mean": mean, "scale": scale, "with_mean": with_mean, "with_std": with_std}
        return Z, params
    return Z


def minmax_scale(
    X: ArrayLike,
    *,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Scale each feature to a given range (default [0, 1])."""
    Xf = _as_2d_float(X, name="X")

    a, b = float(feature_range[0]), float(feature_range[1])
    if not a < b:
        raise ValueError("feature_range must satisfy (min < max).")

    data_min = Xf.min(axis=0)
    data_max = Xf.max(axis=0)
    data_range = data_max - data_min
    scale = np.where(data_range == 0, 1.0, data_range)

    X01 = (Xf - data_min) / scale
    Xs = X01 * (b - a) + a

    if return_params:
        params = {
            "data_min": data_min,
            "data_max": data_max,
            "scale": scale,
            "feature_range": (a, b),
        }
        return Xs, params
    return Xs


def maxabs_scale(
    X: ArrayLike,
    *,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Scale each feature by its maximum absolute value."""
    Xf = _as_2d_float(X, name="X")
    scale = np.max(np.abs(Xf), axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    Xs = Xf / scale
    if return_params:
        return Xs, {"scale": scale}
    return Xs


def l2_normalize_rows(X: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row; rows with ~0 norm remain all-zeros."""
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    Xf = _as_2d_float(X, name="X")
    norms = np.linalg.norm(Xf, axis=1)
    out = Xf.copy()
    mask = norms > eps
    out[mask] = out[mask] / norms[mask, None]
    return out


def _stratified_test_counts(y: np.ndarray, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (classes, n_test_per_class) with sane bounds."""
    classes, counts = np.unique(y, return_counts=True)
    raw = counts * float(test_size)
    # start with rounding
    n_test = np.rint(raw).astype(int)
    # keep at least 1 in each split when possible
    n_test = np.clip(n_test, 1, np.maximum(1, counts - 1))

    # adjust to match the global requested count
    total = int(np.rint(len(y) * float(test_size)))
    total = max(1, min(len(y) - 1, total))
    diff = total - int(n_test.sum())
    if diff != 0:
        # distribute based on fractional parts (largest remainder method)
        frac = raw - np.floor(raw)
        order = np.argsort(frac)[::-1]
        i = 0
        while diff != 0 and i < len(order) * 2:
            k = order[i % len(order)]
            if diff > 0 and n_test[k] < counts[k] - 1:
                n_test[k] += 1
                diff -= 1
            elif diff < 0 and n_test[k] > 1:
                n_test[k] -= 1
                diff += 1
            i += 1
    return classes, n_test


def train_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
):
    """
    Split arrays or matrices into train and test subsets.

    If y is omitted, returns (X_train, X_test).
    """
    rs = _validate_random_state(random_state)
    Xf = _as_2d_float(X, name="X")
    n = Xf.shape[0]

    if not (0 < float(test_size) < 1):
        raise ValueError("test_size must be a float in (0, 1).")

    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = np.asarray(y_arr).reshape(-1)
        if len(y_arr) != n:
            raise ValueError("X and y must have the same number of samples.")
    else:
        y_arr = None

    if stratify is not None and y_arr is None:
        raise ValueError("stratify requires y.")
    strat_arr = np.asarray(stratify) if stratify is not None else None
    if strat_arr is not None and len(strat_arr) != n:
        raise ValueError("stratify must have the same length as X.")

    n_test = int(np.rint(n * float(test_size)))
    n_test = max(1, min(n - 1, n_test))
    n_train = n - n_test

    if not shuffle:
        train_idx = np.arange(0, n_train)
        test_idx = np.arange(n_train, n)
    else:
        rng = np.random.default_rng(rs)
        if strat_arr is None:
            perm = rng.permutation(n)
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]
        else:
            classes, n_test_per = _stratified_test_counts(strat_arr, float(test_size))
            test_parts = []
            train_parts = []
            for c, k in zip(classes, n_test_per):
                idx_c = np.flatnonzero(strat_arr == c)
                rng.shuffle(idx_c)
                test_parts.append(idx_c[:k])
                train_parts.append(idx_c[k:])
            test_idx = np.concatenate(test_parts)
            train_idx = np.concatenate(train_parts)
            rng.shuffle(test_idx)
            rng.shuffle(train_idx)

    X_train, X_test = Xf[train_idx], Xf[test_idx]
    if y_arr is None:
        return X_train, X_test
    return X_train, X_test, y_arr[train_idx], y_arr[test_idx]


def train_val_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    val_size: float = 0.2,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
):
    """Split into train/val/test; if y omitted returns only X parts."""
    rs = _validate_random_state(random_state)
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be >= 0.")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")

    # first: split off the test set
    first = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=stratify if y is not None else None,
        random_state=rs,
    )

    if y is None:
        X_trainval, X_test = first
        y_trainval = None
        y_test = None
    else:
        X_trainval, X_test, y_trainval, y_test = first

    # second: split train/val from trainval
    rel_val = 0.0 if (1 - test_size) == 0 else (val_size / (1 - test_size))
    second_rs = None if rs is None else rs + 1
    second = train_test_split(
        X_trainval,
        y_trainval,
        test_size=rel_val,
        shuffle=shuffle,
        stratify=y_trainval if (y_trainval is not None and stratify is not None) else None,
        random_state=second_rs,
    )

    if y is None:
        X_train, X_val = second
        return X_train, X_val, X_test

    X_train, X_val, y_train, y_val = second
    return X_train, X_val, X_test, y_train, y_val, y_test
