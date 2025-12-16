# Regression Trees

This folder contains a notebook demonstration of **regression trees**: tree-based models that predict continuous values by recursively partitioning the feature space.

---

## Design Philosophy

- **Variance reduction intuition**: splits aim to make targets more homogeneous within leaves
- **Interpretability**: understand predictions as averages within partitions
- **End-to-end clarity**: train → predict → evaluate

---

## Contents

```text
Regression Trees/
├── regression_trees_example.ipynb
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

Open `regression_trees_example.ipynb` and run all cells.

---

## What This Demonstrates

- How regression trees partition input space to reduce within-node error
- How depth controls fit vs generalization
- Evaluating regression performance with metrics like MSE and \(R^2\)

---

## Limitations

- Educational-scale example (not optimized)
- Trees can overfit without careful stopping/pruning

---

## Conclusion

This notebook provides a clear regression tree walkthrough for building intuition about tree-based regression models.
