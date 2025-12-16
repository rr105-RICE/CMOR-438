# rice_ml

`rice_ml` is a **from-scratch machine learning library** designed for educational and exploratory purposes. It provides clean, readable implementations of core machine learning algorithms across **supervised learning** and **data preprocessing**, with an accompanying set of **unsupervised learning notebooks** for concept demos.

This package is intended for learning, experimentation, and deeper understanding of how machine learning algorithms work internally—without relying on black-box frameworks.

---

## Design Philosophy

The guiding principles of `rice_ml` are:

- **From-scratch implementations** (NumPy-based, minimal abstraction)
- **Mathematical transparency** over performance optimization
- **Modular structure** that mirrors standard ML taxonomy
- **Readable code** suitable for coursework and self-study
- **Explicit assumptions and limitations**

This library is not intended to replace production ML frameworks, but to **explain them**.

---

## Package Structure

```text
src/rice_ml/
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

Notebooks are organized separately under `Notebooks/` (see `Notebooks/Supervised_Leaerning/` and `Notebooks/Unsupervised_Learning/`).

Each module is self-contained and mirrors the structure and behavior of its theoretical counterpart.

---

## Processing

### `processing`

Utilities for preparing data before and after modeling.

**Includes:**
- Feature standardization and normalization
- Common preprocessing transformations (scaling, train/test splitting)
- Post-processing helpers for model outputs (accuracy, MSE, R², confusion matrix)

These utilities are intentionally minimal and designed to expose how preprocessing affects downstream algorithms.

---

## Supervised Learning

### `supervised_learning`

Implements classic supervised learning algorithms where labeled data is available.

**Algorithms included (in the installable package):**

- **Linear Regression**  
  Closed-form and gradient-based solutions for regression tasks.

- **Logistic Regression**  
  Binary classification using sigmoid activation and log-loss.

- **Gradient Descent**  
  Generic optimization routines shared across models.

- **k-Nearest Neighbors (kNN)**  
  Distance-based classification with customizable metrics.

- **Distance Metrics**  
  Euclidean and related distance functions used across models.

- **Decision Trees**  
  Tree-based models using greedy recursive splitting.

- **Ensemble Methods**  
  Foundations for combining multiple weak learners.

Additional supervised topics (e.g., perceptron/MLP/regression trees) are demonstrated in the notebooks and may be promoted into the package module set over time.

---

## Unsupervised Learning

### Notebooks (`Notebooks/Unsupervised_Learning`)

The repository also includes unsupervised learning implementations and walkthroughs **as notebooks**, focused on discovery of structure **without labels**.

**Topics included:**

- **K-Means Clustering**  
  Centroid-based clustering using Euclidean distance.

- **DBSCAN**  
  Density-based clustering with explicit noise detection.

- **Principal Component Analysis (PCA)**  
  Variance-based dimensionality reduction via eigen-decomposition.

- **Community Detection (Label Propagation)**  
  Graph-based clustering using connectivity and local consensus.

These methods illustrate different definitions of similarity: **distance, density, variance, and connectivity**.

---

## Educational Focus

The `rice_ml` package is especially suited for:

- Machine learning coursework  
- Algorithm walkthroughs and demos  
- Understanding tradeoffs and assumptions  
- Debugging intuition for real-world ML tools  
- Connecting math to code  

Each algorithm is implemented in a way that mirrors its mathematical definition as closely as possible.

---

## Limitations

- Not optimized for large-scale or production use  
- Limited numerical stability safeguards  
- Focuses on clarity rather than speed  
- Assumes clean, well-structured input data  

These tradeoffs are intentional and aligned with the library’s educational goals.

---

## Conclusion

`rice_ml` provides a cohesive, transparent collection of machine learning building blocks implemented from first principles.

By organizing supervised learning and preprocessing under a single consistent interface—and pairing it with a set of unsupervised learning notebooks—the repo serves as a practical companion for understanding **how machine learning works**.
