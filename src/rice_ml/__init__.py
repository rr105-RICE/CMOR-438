"""
Top-level public API for the rice_ml package.

The unit tests expect common utilities to be importable directly from `rice_ml`.
"""

from .processing.preprocessing import (
    l2_normalize_rows,
    maxabs_scale,
    minmax_scale,
    standardize,
    train_test_split,
    train_val_test_split,
)
from .processing.post_processing import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mae,
    mse,
    precision_score,
    r2_score,
    recall_score,
    rmse,
    roc_auc_score,
)
from .supervised_learning.distance_metrics import euclidean_distance, manhattan_distance

__all__ = [
    # preprocessing
    "standardize",
    "minmax_scale",
    "maxabs_scale",
    "l2_normalize_rows",
    "train_test_split",
    "train_val_test_split",
    # distances
    "euclidean_distance",
    "manhattan_distance",
    # classification metrics
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "roc_auc_score",
    "log_loss",
    # regression metrics
    "mse",
    "rmse",
    "mae",
    "r2_score",
]
