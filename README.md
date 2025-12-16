# rice_ml

`rice_ml` is a **from-scratch machine learning library** designed for educational and exploratory purposes. It provides clean, readable implementations of core machine learning utilities and algorithms (NumPy-based), with an emphasis on mathematical clarity and algorithmic transparency.

This repository also includes **Jupyter notebook examples** that demonstrate additional supervised and unsupervised learning workflows.

---

## Design Philosophy

The guiding principles of `rice_ml` are:

- **From-scratch implementations** (NumPy-based, minimal abstraction)
- **Mathematical transparency** over performance optimization
- **Modular structure** that mirrors standard ML taxonomy (as the codebase grows)
- **Readable code** suitable for coursework and self-study
- **Explicit assumptions and limitations**

This library is not intended to replace production ML frameworks, but to **explain them**.

---

## Repository Structure

```text
.
├── src/
│   └── rice_ml/
│       ├── processing/
│       │   ├── preprocessing.py
│       │   └── post_processing.py
│       ├── supervised_learning/
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── gradient_descent.py
│       │   ├── knn.py
│       │   ├── distance_metrics.py
│       │   ├── decision_tree.py
│       │   └── ensemble_methods.py
│       └── __init__.py
│
├── Notebooks/
│   ├── Supervised_Leaerning/
│   │   └── ... (topic folders + notebooks)
│   └── Unsupervised_Learning/
│       └── ... (topic folders + notebooks)
│
└── tests/
    └── unit/
        └── ... (pytest unit tests)
```

Notes:
- The **Python package** lives in `src/rice_ml/`.
- Some topics (e.g., PCA / K-means / DBSCAN / Perceptron / MLP / Regression Trees) are currently demonstrated in **notebooks** under `Notebooks/` even if there is not yet a corresponding `src/rice_ml/...` module.

---

## Processing

### `processing`

Utilities for preparing data before and after modeling.

**Includes:**
- Feature standardization and normalization
- Preprocessing transformations (scaling, train/test splitting, train/val/test splitting)
- Post-processing metrics (accuracy, precision/recall/F1, log loss, ROC AUC, MSE/RMSE/MAE, \(R^2\), confusion matrices)

These utilities are intentionally minimal and designed to expose how preprocessing affects downstream algorithms.

---

## Supervised Learning

### `supervised_learning`

Implements classic supervised learning algorithms where labeled data is available.

**Algorithms included (in `src/rice_ml/supervised_learning/`):**

- **Linear Regression**
- **Logistic Regression**
- **Gradient Descent** (shared optimization routines)
- **k-Nearest Neighbors (kNN)** (classifier + regressor)
- **Distance Metrics** (Euclidean, Manhattan)
- **Decision Trees** (classification)
- **Ensemble Methods** (foundations for combining learners)

---

## Unsupervised Learning (Notebooks)

The repository includes unsupervised-learning demonstrations in `Notebooks/Unsupervised_Learning/` (clustering, dimensionality reduction, and community detection). These are currently **notebook-first** examples.

---

## Educational Focus

The `rice_ml` package + notebooks are suited for:

- Machine learning coursework
- Algorithm walkthroughs and demos
- Understanding tradeoffs and assumptions
- Debugging intuition for real-world ML tools
- Connecting math to code

---

## Limitations

- Not optimized for large-scale or production use
- Limited numerical stability safeguards
- Focuses on clarity rather than speed
- Assumes clean, well-structured input data

These tradeoffs are intentional and aligned with the library’s educational goals.

---

## Getting Started

Install in editable mode and run tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m pytest -q
```

---

## Quick Usage

```python
import numpy as np
from rice_ml.supervised_learning.knn import KNNClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 1, 1])

clf = KNNClassifier(n_neighbors=3).fit(X, y)
print(clf.predict([[0.2, 0.1], [0.9, 0.8]]))
```

---

## Notebooks

Start here:
- `Notebooks/Supervised_Leaerning/README.md`
- `Notebooks/Unsupervised_Learning/README.md`

Each notebook folder contains a README with purpose, files, and run instructions.

---

## Conclusion

`rice_ml` provides a cohesive, transparent collection of machine learning utilities and algorithms implemented from first principles, plus notebooks that demonstrate how these ideas work in practice.
