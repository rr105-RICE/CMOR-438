# Linear Regression

This folder contains a notebook demonstration of **linear regression** for supervised learning regression tasks.

The notebook focuses on fitting a linear model, interpreting coefficients, and evaluating predictions using regression metrics.

---

## Design Philosophy

- Keep the model **close to the math** (linear predictor + loss)
- Emphasize **interpretability** (coefficients and residuals)
- Use standard metrics to connect predictions to performance

---

## Contents

```text
Linear_Regression/
├── linear_regression_boston.ipynb
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

Open `linear_regression_boston.ipynb` and run all cells.

---

## What This Demonstrates

- Training a linear regression model and generating predictions
- Understanding error through residuals and \(R^2\)
- How feature scaling and train/test splitting affect outcomes

---

## Limitations

- Educational-scale example (not tuned for large datasets)
- Assumes numeric features and relatively clean inputs

---

## Conclusion

This notebook provides an end-to-end linear regression walkthrough and complements the reusable implementation in `src/rice_ml/supervised_learning/linear_regression.py`.
