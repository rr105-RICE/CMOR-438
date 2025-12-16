# rice_ml — Unsupervised Learning Notebooks

This folder contains **unsupervised learning** walkthroughs (clustering, dimensionality reduction, and graph/community structure) implemented as Jupyter notebooks.

The emphasis is on illustrating *what “structure” means without labels* and how different methods operationalize similarity via **distance, density, variance, and connectivity**.

---

## Design Philosophy

- **From-scratch intuition**: highlight the core mechanics (not library magic)
- **Interpretability**: show intermediate artifacts (clusters, components, communities)
- **Exploration-oriented**: encourage parameter sweeps and visual checks

---

## Folder Structure

```text
Notebooks/Unsupervised_Learning/
├── Community Detection/
├── DBScan/
├── K-Means Clustering/
└── PCA/
```

Each subfolder contains a notebook and a short README describing what it demonstrates.

---

## Topics Included

- **K-Means Clustering**  
  Centroid-based clustering and sensitivity to the choice of \(k\).

- **DBSCAN**  
  Density-based clustering with explicit handling of noise points.

- **Principal Component Analysis (PCA)**  
  Dimensionality reduction and visualization via variance maximization / eigen-decomposition.

- **Community Detection (Label Propagation)**  
  Graph-based clustering based on connectivity and local consensus.

---

## How to Run

From the repo root, create an environment and install the package (for shared helpers where applicable):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open any `.ipynb` and run the cells top-to-bottom.

---

## Limitations

- Not tuned for performance or large-scale datasets
- Some notebooks rely on plotting/visual inspection for interpretation
- Focus is on concepts and mechanics rather than deployment

