# Regression Trees (Notebook)

This notebook demonstrates **tree-based regression**, focusing on how recursive partitioning can approximate nonlinear functions and how split criteria differ from classification trees.

---

## Design Philosophy

- **Interpretability**: connect splits to piecewise predictions
- **Mechanics-first**: show variance reduction intuition
- **Workflow clarity**: fit → predict → evaluate

---

## Files

- **Notebook**: `regression_trees_example.ipynb`

---

## What this notebook covers

- Regression-tree splitting intuition (variance reduction)  
- Predicting with piecewise-constant leaf values  
- Evaluation and overfitting behavior as depth increases

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `regression_trees_example.ipynb` in Jupyter and run all cells.

---

## Limitations

- Deep trees can overfit quickly without constraints/pruning
- Greedy splitting can be unstable to small data changes

