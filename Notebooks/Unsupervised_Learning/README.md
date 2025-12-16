# Unsupervised Learning (Notebooks)

This folder contains **unsupervised learning** notebooks (clustering, dimensionality reduction, and graph/community structure). These examples are designed to highlight different definitions of “structure” in data: **distance, density, variance, and connectivity**.

---

## Design Philosophy

- **Concept-first notebooks** that make each algorithm’s assumptions explicit
- **Step-by-step workflows** (fit → inspect → interpret)
- **Readable code** intended for learning rather than production use

---

## Folder Structure

```text
Unsupervised_Learning/
├── Community Detection/
│   ├── community_detection_example.ipynb
│   └── README.md
├── DBScan/
│   ├── dbscan_example.ipynb
│   └── README.md
├── K-Means Clustering/
│   ├── k_means_clustering.ipynb
│   └── README.md
└── PCA/
    ├── pca_example.ipynb
    └── README.md
```

---

## Topics Included

- **K-Means Clustering**: centroid-based clustering using distance
- **DBSCAN**: density-based clustering with noise detection
- **PCA**: variance-based dimensionality reduction
- **Community Detection**: graph-based clustering using connectivity/local consensus

---

## How to Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Open any `.ipynb` in Jupyter and run the cells top-to-bottom.

---

## Limitations

- Examples are designed for small/medium datasets
- Focus is on interpretability and learning, not speed

---

## Conclusion

These notebooks provide hands-on intuition for unsupervised learning methods that discover structure without labels.
