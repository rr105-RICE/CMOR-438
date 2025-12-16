# Perceptron (Notebook)

This notebook demonstrates the **Perceptron** algorithm as a from-scratch, mistake-driven method for **linear binary classification**. The focus is on how weight updates emerge directly from misclassifications and how separability assumptions affect convergence.

---

## Design Philosophy

- **Transparent mechanics**: show updates step-by-step
- **Math-to-code alignment**: keep the implementation close to the rule
- **Exploration-friendly**: encourage changing learning rate / epochs and observing behavior

---

## Files

- **Notebook**: `perceptron_example.ipynb`

---

## What this notebook covers

- The perceptron update rule and decision boundary intuition  
- Convergence behavior under linear separability assumptions  
- The effect of learning rate, epochs, and feature scaling on training

---

## How to run

From the repo root (recommended so imports work consistently):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `perceptron_example.ipynb` in Jupyter and run all cells.

---

## Limitations

- Designed for learning, not for production-scale training
- Behavior depends strongly on feature scaling and separability

