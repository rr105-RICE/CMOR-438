# Perceptron

This folder contains a notebook demonstration of the **Perceptron** algorithm: a classic linear classifier trained by mistake-driven updates.

---

## Design Philosophy

- **Mistake-driven learning**: updates happen only on misclassifications
- **Math-first implementation**: the update rule is visible and inspectable
- **Interpretability**: weights correspond directly to feature influence in a linear decision boundary

---

## Contents

```text
 Perceptron/
├── perceptron_example.ipynb
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

Open `perceptron_example.ipynb` and run all cells.

---

## What This Demonstrates

- How linear separability affects convergence
- How weight updates shift the decision boundary
- Why the Perceptron is a foundational idea for more complex neural models

---

## Limitations

- Only handles linearly separable problems well
- Sensitive to feature scaling and learning-rate-like choices (if included)
- Notebook is for intuition, not production training

---

## Conclusion

This notebook provides a clear Perceptron walkthrough for building intuition about linear classification and learning by updates.
