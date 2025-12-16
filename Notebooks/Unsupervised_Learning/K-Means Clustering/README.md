# K-Means Clustering (Notebook)

This notebook demonstrates **k-means clustering**, focusing on centroid updates, assignment steps, and how the choice of \(k\) and initialization affects results.

---

## Design Philosophy

- **Centroid intuition**: make the assignment/update loop explicit
- **Exploration encouraged**: vary \(k\), initialization, and iterations
- **Visual interpretation**: connect clusters to geometry in feature space

---

## Files

- **Notebook**: `k_means_clustering.ipynb`

---

## What this notebook covers

- The k-means objective and alternating minimization steps  
- Choosing \(k\) (intuition and simple heuristics)  
- Common failure modes (local minima, scaling sensitivity)

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `k_means_clustering.ipynb` in Jupyter and run all cells.

---

## Limitations

- Requires choosing \(k\) in advance
- Assumes roughly spherical clusters under Euclidean distance
- Sensitive to feature scaling and initialization

