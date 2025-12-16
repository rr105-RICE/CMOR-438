# Ensemble Methods

This folder contains a notebook demonstration of **ensemble methods**: combining multiple models to improve performance and/or stability compared to a single learner.

---

## Design Philosophy

- Focus on **why ensembles work** (variance reduction, stability, error cancellation)
- Keep the workflow **explicit**: train components → combine → evaluate
- Emphasize interpretation of results rather than framework-specific APIs

---

## Contents

```text
Ensemble_Methods/
├── ensemble_methods_example.ipynb
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

Open `ensemble_methods_example.ipynb` and run all cells.

---

## What This Demonstrates

- The motivation for combining learners (bias/variance intuition)
- Comparing ensemble behavior against individual models
- Using standard evaluation metrics to judge improvements

---

## Limitations

- Notebook is designed for educational scale datasets
- Focus is conceptual clarity rather than tuned, production-grade ensembles

---

## Conclusion

This notebook provides an accessible introduction to ensemble thinking and pairs with the reusable code in `src/rice_ml/supervised_learning/ensemble_methods.py`.
