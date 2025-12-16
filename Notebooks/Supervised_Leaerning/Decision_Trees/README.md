# Decision Trees (Notebook)

This notebook demonstrates a **decision tree** workflow for classification, focusing on *how greedy splitting decisions are made* and how model complexity affects generalization.

---

## Design Philosophy

- **Algorithmic transparency** over performance tricks
- **Interpretability first**: connect splits to decision logic
- **Hands-on experimentation**: vary depth/criteria and observe behavior

---

## Files

- **Notebook**: `decision_tree_ex.ipynb`

---

## What this notebook covers

- Greedy recursive splitting and impurity intuition  
- Training vs. evaluation workflow and common failure modes (overfitting)  
- The impact of depth/leaf constraints on accuracy and stability

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `decision_tree_ex.ipynb` and run the cells top-to-bottom.

---

## Limitations

- Not optimized for large datasets
- Greedy splitting can be unstable (small data changes can alter the tree)

