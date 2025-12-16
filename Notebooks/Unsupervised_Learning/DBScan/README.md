# DBSCAN (Notebook)

This notebook demonstrates **DBSCAN**, a density-based clustering method that can discover arbitrarily shaped clusters and explicitly label **noise/outliers**.

---

## Design Philosophy

- **Density intuition**: make “core/border/noise” concrete
- **Parameter exploration**: observe how `eps` and `min_samples` reshape clusters
- **Visualization oriented**: interpret clustering via plots where possible

---

## Files

- **Notebook**: `dbscan_example.ipynb`

---

## What this notebook covers

- Core points, border points, and noise definitions  
- The roles of `eps` and `min_samples` and common tuning pitfalls  
- Strengths/weaknesses vs. centroid-based methods like k-means

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `dbscan_example.ipynb` in Jupyter and run all cells.

---

## Limitations

- Parameter sensitivity (especially in varying-density datasets)
- Distance scaling matters (standardization can be important)

