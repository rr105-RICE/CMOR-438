# Supervised Learning (Notebooks)

This folder contains **supervised learning** notebooks (classification/regression). These examples are intended for learning and experimentation and are written to make the algorithmic steps easy to follow.

---

## Design Philosophy

- **Transparent workflows** that mirror the math and algorithm steps
- **Minimal “magic”**: explicit preprocessing, training, prediction, and evaluation
- **Readable cells** suitable for coursework and self-study

---

## Folder Structure

```text
Supervised_Leaerning/
├──  Perceptron/
│   ├── perceptron_example.ipynb
│   └── README.md
├── Decision_Trees/
│   ├── decision_tree_ex.ipynb
│   └── README.md
├── Ensemble_Methods/
│   ├── ensemble_methods_example.ipynb
│   └── README.md
├── Gradient_Descent/
│   ├── Gradient_Descent,note.ipynb
│   └── README.md
├── KNN/
│   ├── KNN_example.ipynb
│   ├── iris.csv
│   └── README.md
├── Linear_Regression/
│   ├── linear_regression_boston.ipynb
│   └── README.md
├── Logistic Regression/
│   ├── logistic_regression_pima.ipynb
│   └── README.md
├── Multilayer Perceptron/
│   ├── multilayer_perceptron_example.ipynb
│   └── README.md
└── Regression Trees/
    ├── regression_trees_example.ipynb
    └── README.md
```

---

## Topics Included

- **Perceptron**: linear classification with mistake-driven updates
- **Decision Trees**: greedy recursive splitting (classification)
- **Ensemble Methods**: combining multiple learners (conceptual + practical)
- **Gradient Descent**: step-by-step optimization intuition
- **kNN**: distance-based classification on a labeled dataset (`iris.csv`)
- **Linear Regression**: regression fitting + evaluation
- **Logistic Regression**: binary classification with probabilities + metrics
- **MLP**: simple neural network workflow (forward + training loop)
- **Regression Trees**: tree-based regression concepts and evaluation

---

## How to Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Then open any `.ipynb` in Jupyter (VS Code, JupyterLab, or classic notebook) and run cells top-to-bottom.

---

## Limitations

- Not optimized for large-scale data
- Assumes relatively clean inputs
- Focuses on clarity over performance

---

## Conclusion

These notebooks provide a practical companion to the `rice_ml` package by showing supervised learning algorithms and workflows end-to-end.
