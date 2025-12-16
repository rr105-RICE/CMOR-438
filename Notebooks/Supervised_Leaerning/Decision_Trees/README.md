# Decision Trees (Classification)

This folder contains a notebook that demonstrates **decision tree classification**: greedy recursive splitting, interpreting splits, and evaluating classification performance.

---

## Design Philosophy

- **Algorithmic transparency**: show how splits reduce impurity
- **Readable model behavior**: interpret predictions by traversing the tree
- **Minimal abstraction**: keep the workflow close to the theory

---

## Contents

```text
Decision_Trees/
├── decision_tree_ex.ipynb
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

Then open `decision_tree_ex.ipynb` and run all cells.

---

## What This Demonstrates

- Greedy splitting and why different thresholds change the tree structure
- How trees trade interpretability for overfitting risk (depth vs generalization)
- End-to-end evaluation using standard classification metrics

---

## Limitations

- Demonstration focuses on clarity, not computational efficiency
- Small datasets and simple feature sets are assumed

---

## Conclusion

This notebook provides hands-on intuition for decision tree classifiers and complements the reusable decision tree implementation in `src/rice_ml/supervised_learning/decision_tree.py`.
