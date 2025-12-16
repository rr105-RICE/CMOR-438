from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["PCA"]


def _as2d_float(X, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if arr.size == 0 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


@dataclass
class PCA:
    """
    Principal Component Analysis (PCA) via covariance eigen-decomposition.

    sklearn-like attributes after fit:
    - mean_
    - components_
    - explained_variance_
    - explained_variance_ratio_
    """

    n_components: int

    mean_: Optional[np.ndarray] = None
    components_: Optional[np.ndarray] = None
    explained_variance_: Optional[np.ndarray] = None
    explained_variance_ratio_: Optional[np.ndarray] = None

    def __post_init__(self):
        if int(self.n_components) <= 0:
            raise ValueError("n_components must be positive.")
        self.n_components = int(self.n_components)

    def __repr__(self) -> str:  # exact repr required by tests
        return f"PCA(n_components={self.n_components})"

    def fit(self, X):
        X = _as2d_float(X, name="X")
        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError("n_components cannot exceed number of features.")

        mean = X.mean(axis=0)
        Xc = X - mean

        # covariance matrix (features x features)
        denom = max(n_samples - 1, 1)
        cov = (Xc.T @ Xc) / float(denom)

        # eigh for symmetric matrices; returns ascending eigenvalues
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        comps = eigvecs[:, : self.n_components].T  # (n_components, n_features)
        explained = eigvals[: self.n_components].astype(float, copy=False)

        total = float(np.sum(eigvals))
        ratio = explained / total if total > 0 else np.zeros_like(explained)

        self.mean_ = mean
        self.components_ = comps
        self.explained_variance_ = explained
        self.explained_variance_ratio_ = ratio
        return self

    def transform(self, X):
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Call fit before transform.")
        X = _as2d_float(X, name="X")
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Call fit before inverse_transform.")
        Z = np.asarray(X_transformed, dtype=float)
        if Z.ndim != 2:
            raise ValueError("X must be 2D.")
        return Z @ self.components_ + self.mean_

