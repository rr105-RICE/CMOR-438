from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["DBSCAN"]


def _as2d_float(X, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    # (n, d) -> (n, n) euclidean
    x2 = np.sum(X * X, axis=1)
    d2 = np.maximum(x2[:, None] + x2[None, :] - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


@dataclass
class DBSCAN:
    """
    Minimal DBSCAN implementation with fit / fit_predict.

    Labels:
    -1 indicates noise.
    0..k-1 are cluster ids.
    """

    eps: float = 0.5
    min_samples: int = 5

    labels_: Optional[np.ndarray] = None

    def __post_init__(self):
        if float(self.eps) <= 0:
            raise ValueError("eps must be > 0.")
        if int(self.min_samples) <= 0:
            raise ValueError("min_samples must be > 0.")
        self.eps = float(self.eps)
        self.min_samples = int(self.min_samples)

    def fit(self, X):
        X = _as2d_float(X, name="X")
        n = X.shape[0]
        D = _pairwise_distances(X)

        neighbors = [np.flatnonzero(D[i] <= self.eps) for i in range(n)]
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neigh = neighbors[i]
            if len(neigh) < self.min_samples:
                labels[i] = -1
                continue

            # start a new cluster
            labels[i] = cluster_id
            seeds = list(neigh)
            j = 0
            while j < len(seeds):
                p = seeds[j]
                if not visited[p]:
                    visited[p] = True
                    neigh_p = neighbors[p]
                    if len(neigh_p) >= self.min_samples:
                        # expand
                        for q in neigh_p:
                            if q not in seeds:
                                seeds.append(int(q))
                if labels[p] == -1:
                    labels[p] = cluster_id
                j += 1

            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

