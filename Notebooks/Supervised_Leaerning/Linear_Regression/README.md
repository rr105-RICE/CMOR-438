# Linear Regression (Notebook)

This notebook demonstrates **linear regression** as a baseline supervised learning method for regression tasks, focusing on model fit, interpretation, and error metrics.

---

## Design Philosophy

- **Math-first clarity**: connect coefficients to loss minimization
- **End-to-end workflow**: fit → predict → evaluate
- **Interpretability**: understand what the model is actually doing

---

## Files

- **Notebook**: `linear_regression_boston.ipynb`

---

## What this notebook covers

- Linear model assumptions and coefficient interpretation  
- Training a regression model and generating predictions  
- Evaluation with metrics such as MSE and \(R^2\)

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `linear_regression_boston.ipynb` in Jupyter and run the cells top-to-bottom.

---

## Limitations

- Linear regression can underfit nonlinear relationships
- Sensitive to outliers and feature scaling/conditioning

