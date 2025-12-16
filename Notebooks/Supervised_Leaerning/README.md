# rice_ml — Supervised Learning Notebooks

This folder contains **supervised learning** walkthroughs implemented as Jupyter notebooks. The notebooks emphasize **from-scratch reasoning** and **algorithm transparency**, pairing mathematical intuition with runnable code.

These are intended for learning and experimentation rather than production modeling.

---

## Design Philosophy

- **Educational first**: clarity over optimization
- **Algorithmic transparency**: show intermediate steps and assumptions
- **Hands-on exploration**: encourage changing hyperparameters and observing behavior
- **Minimal reliance on black-box frameworks**

---

## Folder Structure

```text
Notebooks/Supervised_Leaerning/
├──  Perceptron/
├── Decision_Trees/
├── Ensemble_Methods/
├── Gradient_Descent/
├── KNN/
├── Linear_Regression/
├── Logistic Regression/
├── Multilayer Perceptron/
└── Regression Trees/
```

Each subfolder contains a notebook and a short README describing what it demonstrates.

---

## Topics Included

- **Perceptron**  
  Mistake-driven updates for a linear binary classifier.

- **Decision Trees**  
  Greedy recursive splitting for classification.

- **Ensemble Methods**  
  Combining multiple learners and comparing to single-model baselines.

- **Gradient Descent**  
  Step-by-step optimization intuition on simple objectives.

- **k-Nearest Neighbors (kNN)**  
  Distance-based classification and sensitivity to \(k\) / distance choices.

- **Linear Regression**  
  Regression fit + evaluation on a classic dataset workflow.

- **Logistic Regression**  
  Binary classification with probabilities and evaluation metrics.

- **Multilayer Perceptron (MLP)**  
  Feedforward network intuition and training workflow.

- **Regression Trees**  
  Tree-based regression with interpretability and error analysis.

---

## How to Run

From the repo root, create an environment and install the package (so notebooks can import `rice_ml` where applicable):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open any `.ipynb` (VS Code, JupyterLab, or classic notebook) and run cells top-to-bottom.

---

## Limitations

- Not optimized for large datasets
- Some notebooks assume local relative paths and bundled datasets
- Focuses on understanding and visualization, not deployment concerns

