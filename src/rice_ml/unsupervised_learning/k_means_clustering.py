from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["KMeans"]


def _as2d_float(X, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if arr.size == 0 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # (n, d) vs (k, d) -> (n, k)
    x2 = np.sum(X * X, axis=1)[:, None]
    c2 = np.sum(centers * centers, axis=1)[None, :]
    return np.maximum(x2 + c2 - 2.0 * X @ centers.T, 0.0)


@dataclass
class KMeans:
    """
    Minimal k-means clustering (NumPy) with a sklearn-like API.

    Attributes set after fit:
    - cluster_centers_: (k, d)
    - labels_: (n,)
    - inertia_: float (sum of squared distances to assigned centroids)
    """

    n_clusters: int = 8
    max_iter: int = 300
    tol: float = 1e-4
    random_state: Optional[int] = None

    cluster_centers_: Optional[np.ndarray] = None
    labels_: Optional[np.ndarray] = None
    inertia_: Optional[float] = None

    def __post_init__(self):
        if int(self.n_clusters) <= 0:
            raise ValueError("n_clusters must be positive.")
        self.n_clusters = int(self.n_clusters)

    def fit(self, X):
        X = _as2d_float(X, name="X")
        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_clusters cannot exceed number of samples.")

        rng = np.random.default_rng(self.random_state)
        init_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centers = X[init_idx].copy()

        labels = None
        inertia = None

        for _ in range(int(self.max_iter)):
            d2 = _squared_distances(X, centers)
            new_labels = np.argmin(d2, axis=1)

            new_centers = np.zeros((self.n_clusters, n_features), dtype=float)
            for k in range(self.n_clusters):
                members = X[new_labels == k]
                if len(members) == 0:
                    # re-seed empty cluster to a random point
                    new_centers[k] = X[rng.integers(0, n_samples)]
                else:
                    new_centers[k] = members.mean(axis=0)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            labels = new_labels
            inertia = float(np.sum(_squared_distances(X, centers)[np.arange(n_samples), labels]))
            if shift <= float(self.tol):
                break

        self.cluster_centers_ = centers
        self.labels_ = labels.astype(int, copy=False)
        self.inertia_ = float(inertia)
        return self

    def predict(self, X):
        if self.cluster_centers_ is None or self.labels_ is None:
            raise RuntimeError("Call fit before predict.")
        X = _as2d_float(X, name="X")
        d2 = _squared_distances(X, self.cluster_centers_)
        return np.argmin(d2, axis=1).astype(int, copy=False)

