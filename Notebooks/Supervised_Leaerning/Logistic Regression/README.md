# Logistic Regression (Notebook)

This notebook demonstrates **logistic regression** for binary classification, emphasizing probability modeling (via the sigmoid), log-loss optimization, and evaluation beyond raw accuracy.

---

## Design Philosophy

- **Probability-first**: focus on \(P(y=1 \mid x)\), not just labels
- **Transparent optimization**: connect gradients to learning
- **Metric awareness**: understand confusion matrices and tradeoffs

---

## Files

- **Notebook**: `logistic_regression_pima.ipynb`

---

## What this notebook covers

- Sigmoid activation and decision thresholds  
- Training with log-loss and interpreting predicted probabilities  
- Evaluation metrics (accuracy, confusion matrix, related scores)

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `logistic_regression_pima.ipynb` in Jupyter and run all cells.

---

## Limitations

- Assumes a linear decision boundary in feature space
- Sensitive to feature scaling and class imbalance without care

