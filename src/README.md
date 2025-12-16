# rice_ml (Source Code)

`src/` contains the installable Python package for this repository: **`rice_ml`**.

The goal of the source package is to provide **from-scratch, NumPy-first** implementations that prioritize **mathematical clarity** and **readability** over performance, making it suitable for coursework, self-study, and algorithm walkthroughs.

---

## Design Philosophy

- **From-scratch implementations** with minimal abstraction
- **Readable code** that mirrors the math
- **Modular organization** aligned with standard ML taxonomy
- **Explicit assumptions** (inputs are clean, shapes are consistent)

---

## Package Structure

```text
rice_ml/
├── processing/
│   ├── preprocessing.py
│   └── post_processing.py
│
├── supervised_learning/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── gradient_descent.py
│   ├── knn.py
│   ├── distance_metrics.py
│   ├── decision_tree.py
│   └── ensemble_methods.py
│
└── __init__.py
```

---

## Processing

### `processing`

Utilities used before and after modeling.

**Includes:**
- Feature standardization / normalization
- Train/test splitting helpers
- Simple evaluation helpers (e.g., accuracy, MSE, R², confusion matrix)

---

## Supervised Learning

### `supervised_learning`

Implements supervised learning algorithms where labeled data is available.

**Includes:**
- **Linear Regression** (closed-form + gradient-based)
- **Logistic Regression** (binary classification via sigmoid + log-loss)
- **Gradient Descent** (shared optimization routines)
- **kNN** (distance-based classification)
- **Distance Metrics** (shared distance computations)
- **Decision Trees** (greedy recursive splitting)
- **Ensemble Methods** (combining weak learners)

---

## How to use (editable install)

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then import:

```python
from rice_ml.supervised_learning.knn import KNNClassifier
```

---

## Limitations

- Not tuned for performance or large datasets
- Limited numerical-stability hardening
- Assumes inputs are well-formed (dtype/shape)

