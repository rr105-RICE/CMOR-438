# Principal Component Analysis (PCA)

This folder contains a notebook demonstration of **principal component analysis (PCA)** for dimensionality reduction and visualization.

The notebook emphasizes variance maximization, eigen-decomposition intuition, and how projecting onto principal components changes the representation of data.

---

## Design Philosophy

- **Variance-first interpretation**: PCs as directions of maximum variance
- **Math transparency**: covariance/eigendecomposition ideas are explicit
- **Visualization-oriented**: show how PCA helps interpret high-dimensional data

---

## Contents

```text
PCA/
├── pca_example.ipynb
└── README.md
```

---

## How to Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Open `pca_example.ipynb` and run all cells.

---

## What This Demonstrates

- How PCA finds orthogonal directions that capture variance
- How many components are needed to preserve most information
- How PCA can simplify visualization and downstream learning

---

## Limitations

- PCA is linear; it won’t capture non-linear manifolds
- Sensitive to feature scaling (standardize before PCA when appropriate)

---

## Conclusion

This notebook provides an interpretable PCA walkthrough for understanding dimensionality reduction from first principles.
