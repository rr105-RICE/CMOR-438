# Logistic Regression

This folder contains a notebook demonstration of **logistic regression** for binary classification.

The notebook emphasizes predicted probabilities, decision thresholds, and evaluating a classifier with standard metrics.

---

## Design Philosophy

- **Probabilistic interpretation**: logits → sigmoid → probabilities
- **Transparent loss**: log loss (cross-entropy) is explicit
- **End-to-end workflow**: preprocess → train → predict → evaluate

---

## Contents

```text
Logistic Regression/
├── logistic_regression_pima.ipynb
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

Open `logistic_regression_pima.ipynb` and run all cells.

---

## What This Demonstrates

- Training a binary classifier that outputs probabilities
- Using thresholds to convert probabilities into class predictions
- Evaluating performance with metrics like accuracy, precision/recall/F1, and log loss

---

## Limitations

- Focuses on clarity over numerical edge-case handling
- Educational-scale example; not tuned for production performance

---

## Conclusion

This notebook provides a transparent logistic regression workflow and complements the reusable implementation in `src/rice_ml/supervised_learning/logistic_regression.py`.
