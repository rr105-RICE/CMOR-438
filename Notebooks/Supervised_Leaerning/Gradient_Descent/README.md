# Gradient Descent (Notebook)

This notebook is a conceptual walkthrough of **gradient descent**, showing how iterative optimization moves “downhill” on an objective function and why choices like step size matter.

---

## Design Philosophy

- **Mechanics over magic**: emphasize gradients, updates, and convergence
- **Visual intuition**: interpret optimization trajectories where possible
- **Reusable ideas**: connect to how ML models are trained in practice

---

## Files

- **Notebook**: `Gradient_Descent,note.ipynb`

---

## What this notebook covers

- The gradient descent update rule and convergence intuition  
- The role of learning rate (too small vs. too large)  
- Common pitfalls (local minima intuition, slow convergence, scaling)

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `Gradient_Descent,note.ipynb` and run all cells.

---

## Limitations

- Educational examples are simplified relative to real model training
- Numerical behavior depends on scaling and objective conditioning

