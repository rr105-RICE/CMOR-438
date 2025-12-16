# Multilayer Perceptron (MLP) (Notebook)

This notebook demonstrates a **multilayer perceptron (MLP)** workflow for supervised learning, focusing on how forward passes, nonlinear activations, and backpropagation work together during training.

---

## Design Philosophy

- **Mechanics over framework**: emphasize what backprop is doing
- **Math-to-code mapping**: keep the implementation conceptually aligned
- **Experimentation encouraged**: vary depth/width/learning rate and observe changes

---

## Files

- **Notebook**: `multilayer_perceptron_example.ipynb`

---

## What this notebook covers

- Forward pass through layered nonlinear transformations  
- Loss computation and gradient-based updates (backprop intuition)  
- The effect of architecture and hyperparameters on learning

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `multilayer_perceptron_example.ipynb` in Jupyter and run all cells.

---

## Limitations

- Educational implementation (not optimized, limited stability safeguards)
- Training dynamics can be sensitive to initialization and scaling

