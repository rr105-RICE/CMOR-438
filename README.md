# CMOR 438 Data Science & Machine Learning Example Repo

## Purpose

This repository contains a small **Python machine learning package** (`rice_ml`) plus **Jupyter notebook examples** demonstrating common supervised and unsupervised learning techniques.

## What’s included

- **Reusable ML code**: algorithms and utilities in `src/rice_ml/` (e.g., KNN, decision trees, regression/classification helpers).
- **Unit tests**: `pytest` tests in `tests/`.
- **Notebooks**: runnable demos in `Notebooks/` organized by topic.

## Maintainer

- Ricardo Rivera (`rr105@rice.edu`)

## Getting Started

Install in editable mode and run tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## Quick usage (package)

```python
import numpy as np
from rice_ml.supervised_learning.knn import KNNClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 1, 1])

clf = KNNClassifier(n_neighbors=3).fit(X, y)
print(clf.predict([[0.2, 0.1], [0.9, 0.8]]))
```

## Notebooks

See `Notebooks/` for example workflows (each folder contains a short README explaining what it demonstrates).
