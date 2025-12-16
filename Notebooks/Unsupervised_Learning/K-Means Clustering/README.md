# K-Means Clustering

This folder contains a notebook demonstration of **k-means clustering**: a centroid-based method that partitions data into \(k\) groups by minimizing within-cluster distances.

---

## Design Philosophy

- **Centroid intuition**: alternate between assignment and centroid update
- **Transparent objective**: within-cluster sum of squares drives the algorithm
- **Visualization**: cluster assignments and centroids are inspected directly

---

## Contents

```text
K-Means Clustering/
├── k_means_clustering.ipynb
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

Open `k_means_clustering.ipynb` and run all cells.

---

## What This Demonstrates

- The assignment/update loop that defines k-means
- Sensitivity to initialization and how it affects local minima
- Choosing \(k\) (conceptually) and interpreting cluster structure

---

## Limitations

- Assumes roughly spherical clusters under Euclidean distance
- Sensitive to feature scaling and outliers
- Requires choosing \(k\) ahead of time

---

## Conclusion

This notebook provides an interpretable k-means walkthrough and builds intuition for centroid-based clustering.
