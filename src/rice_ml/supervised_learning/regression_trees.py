from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["RegressionTree"]


def _as2d_float(X, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _as1d_float(y, *, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


@dataclass
class _Node:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: Optional[float] = None

    def is_leaf(self) -> bool:
        return self.feature_index is None


@dataclass
class RegressionTree:
    """
    CART-style regression tree with variance reduction splits.

    API required by tests: fit / predict / score and TypeError before fit.
    """

    max_depth: int = 5
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    tree_: Optional[_Node] = None

    def fit(self, X, y):
        X = _as2d_float(X, name="X")
        y = _as1d_float(y, name="y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.tree_ = self._grow(X, y, depth=0)
        return self

    def predict(self, X) -> np.ndarray:
        # tests expect TypeError before fit
        if self.tree_ is None:
            raise TypeError("Model is not fitted yet.")
        X = _as2d_float(X, name="X")
        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            out[i] = self._predict_one(X[i], self.tree_)
        return out

    def score(self, X, y) -> float:
        y_true = _as1d_float(y, name="y")
        y_pred = self.predict(X)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def _predict_one(self, x: np.ndarray, node: _Node) -> float:
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return float(node.value)

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        # stopping conditions
        if (
            depth >= int(self.max_depth)
            or X.shape[0] < int(self.min_samples_split)
            or np.allclose(y, y[0])
        ):
            return _Node(value=float(np.mean(y)))

        feat, thresh, left_idx, right_idx = self._best_split(X, y)
        if feat is None:
            return _Node(value=float(np.mean(y)))

        left = self._grow(X[left_idx], y[left_idx], depth + 1)
        right = self._grow(X[right_idx], y[right_idx], depth + 1)
        return _Node(feature_index=feat, threshold=thresh, left=left, right=right, value=float(np.mean(y)))

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        n_samples, n_features = X.shape
        best_feat = None
        best_thresh = None
        best_score = np.inf
        best_left = None
        best_right = None

        for j in range(n_features):
            xj = X[:, j]
            unique = np.unique(xj)
            if unique.size <= 1:
                continue
            # candidate thresholds between sorted unique values
            unique.sort()
            thresholds = (unique[:-1] + unique[1:]) / 2.0
            for t in thresholds:
                left = xj <= t
                right = ~left
                n_left = int(np.sum(left))
                n_right = n_samples - n_left
                if n_left < int(self.min_samples_leaf) or n_right < int(self.min_samples_leaf):
                    continue

                # weighted SSE (variance * n)
                y_left = y[left]
                y_right = y[right]
                sse = float(np.sum((y_left - np.mean(y_left)) ** 2) + np.sum((y_right - np.mean(y_right)) ** 2))
                if sse < best_score:
                    best_score = sse
                    best_feat = j
                    best_thresh = float(t)
                    best_left = left
                    best_right = right

        return best_feat, best_thresh, best_left, best_right

