# rice_ml (source package)

This directory contains the **installable Python package** `rice_ml` (using a `src/` layout).

The code is intentionally written for **clarity and educational value**: NumPy-based implementations, minimal abstraction, and explicit assumptions.

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

## Design Philosophy

- **From-scratch implementations** (NumPy-based)
- **Mathematical transparency** over performance optimization
- **Readable code** suitable for coursework and self-study
- **Small, composable utilities** (preprocessing + metrics)

---

## How to use

From the repository root:

```bash
pip install -e .[dev]
python -m pytest -q
```

See the repo root `README.md` for a more complete overview and links to the notebooks.
