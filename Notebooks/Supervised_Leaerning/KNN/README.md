# K-Nearest Neighbors (kNN)

This folder contains a notebook demonstration of **k-nearest neighbors (kNN)** for supervised learning, using a small labeled dataset.

The goal is to show how a distance-based learner makes decisions and how hyperparameters like \(k\) and the distance metric affect predictions.

---

## Design Philosophy

- **From-scratch intuition**: emphasize the neighbor lookup + voting idea
- **Mathematical transparency**: distance, neighbor selection, and aggregation are explicit
- **Readable workflow**: data → fit → predict → evaluate

---

## Contents

```text
KNN/
├── KNN_example.ipynb   # main walkthrough notebook
└── iris.csv            # dataset used by the notebook
```

---

## How to Run

1. From the repository root, install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

2. Open `KNN_example.ipynb` and run all cells top-to-bottom.

---

## What This Demonstrates

- How kNN uses **distance** (e.g., Euclidean/Manhattan) to find neighbors
- How \(k\) controls the **bias/variance tradeoff**
- How class predictions can be explained by the **local neighborhood** in feature space

---

## Limitations

- kNN can be slow at prediction time for large datasets (no indexing/acceleration here)
- Results depend heavily on feature scaling and distance choice
- Demonstration assumes relatively clean, numeric feature inputs

---

## Conclusion

This notebook provides an end-to-end, interpretable kNN example and complements the reusable kNN implementation in `src/rice_ml/supervised_learning/knn.py`.
