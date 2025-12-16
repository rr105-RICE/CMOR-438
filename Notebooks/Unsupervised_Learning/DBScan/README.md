# DBSCAN

This folder contains a notebook demonstration of **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).

The notebook highlights how density connectivity forms clusters and how DBSCAN can explicitly label outliers as noise.

---

## Design Philosophy

- **Density intuition**: clusters are dense regions separated by sparse regions
- **Parameter transparency**: show how `eps` and `min_samples` change outcomes
- **Noise-aware clustering**: emphasize outlier handling as a first-class behavior

---

## Contents

```text
DBScan/
├── dbscan_example.ipynb
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

Open `dbscan_example.ipynb` and run all cells.

---

## What This Demonstrates

- Core points vs border points vs noise
- How DBSCAN discovers clusters of arbitrary shape (unlike k-means)
- Why scaling and distance choice can strongly affect DBSCAN

---

## Limitations

- Parameter selection can be challenging without domain insight
- Performance can degrade on very large datasets without acceleration structures

---

## Conclusion

This notebook provides a clear DBSCAN walkthrough for understanding density-based clustering from first principles.
