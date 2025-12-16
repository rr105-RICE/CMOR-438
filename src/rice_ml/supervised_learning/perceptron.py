from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["Perceptron"]


def _as2d_float(X, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _as1d_int(y, *, name: str = "y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr.astype(int, copy=False)


@dataclass
class Perceptron:
    """
    Binary perceptron classifier (labels {0,1}) with sklearn-like API.
    """

    learning_rate: float = 1.0
    max_iter: int = 1000
    random_state: Optional[int] = None
    shuffle: bool = True

    coef_: Optional[np.ndarray] = None
    intercept_: Optional[float] = None

    def fit(self, X, y):
        X = _as2d_float(X, name="X")
        y = _as1d_int(y, name="y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])) and not np.array_equal(classes, np.array([0])) and not np.array_equal(classes, np.array([1])):
            raise ValueError("Perceptron supports only binary labels {0,1}.")

        # map {0,1} -> {-1,+1}
        yt = np.where(y == 1, 1.0, -1.0)
        n_samples, n_features = X.shape

        w = np.zeros(n_features, dtype=float)
        b = 0.0

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n_samples)
        lr = float(self.learning_rate)

        for _ in range(int(self.max_iter)):
            if self.shuffle:
                rng.shuffle(indices)
            errors = 0
            for i in indices:
                if yt[i] * (np.dot(w, X[i]) + b) <= 0:
                    w += lr * yt[i] * X[i]
                    b += lr * yt[i]
                    errors += 1
            if errors == 0:
                break

        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before predict.")
        X = _as2d_float(X, name="X")
        scores = X @ self.coef_ + self.intercept_
        return (scores >= 0).astype(int)

    def score(self, X, y) -> float:
        y = _as1d_int(y, name="y")
        preds = self.predict(X)
        return float(np.mean(preds == y))

