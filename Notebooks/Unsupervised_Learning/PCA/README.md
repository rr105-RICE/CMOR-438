# Principal Component Analysis (PCA) (Notebook)

This notebook demonstrates **PCA** as a variance-maximizing method for **dimensionality reduction** and visualization, connecting the linear algebra (covariance, eigenvectors) to practical transforms.

---

## Design Philosophy

- **Linear algebra clarity**: keep the eigen-decomposition intuition front-and-center
- **Interpretation focused**: relate components to explained variance
- **Visualization oriented**: show how PCA changes the view of data

---

## Files

- **Notebook**: `pca_example.ipynb`

---

## What this notebook covers

- Centering/scaling and why it matters for PCA  
- Covariance structure and principal directions  
- Explained variance and choosing the number of components

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `pca_example.ipynb` in Jupyter and run all cells.

---

## Limitations

- PCA is linear (won’t capture nonlinear manifolds)
- Sensitive to scaling and outliers

