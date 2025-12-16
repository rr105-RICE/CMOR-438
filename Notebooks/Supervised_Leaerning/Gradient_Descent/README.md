# Gradient Descent (Optimization)

This folder contains a notebook walkthrough of **gradient descent**: the core optimization idea behind many machine learning models.

The notebook emphasizes how the update rule follows the gradient, and how learning rate and curvature affect convergence.

---

## Design Philosophy

- **Math-to-code mapping**: connect the update rule directly to implementation
- **Step-by-step reasoning**: visualize/inspect intermediate values
- **Minimal abstraction**: keep the optimization loop explicit

---

## Contents

```text
Gradient_Descent/
├── Gradient_Descent,note.ipynb
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

Open `Gradient_Descent,note.ipynb` and run all cells.

---

## What This Demonstrates

- Why the negative gradient points in the direction of steepest descent
- The effect of step size (learning rate) on stability and speed
- Common failure modes (overshooting, slow convergence)

---

## Limitations

- Examples focus on clarity over numerical sophistication
- Not meant as a production optimizer (no advanced line search, momentum, etc.)

---

## Conclusion

This notebook builds optimization intuition and complements the reusable optimization helpers in `src/rice_ml/supervised_learning/gradient_descent.py`.
