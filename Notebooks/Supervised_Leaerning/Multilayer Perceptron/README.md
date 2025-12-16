# Multilayer Perceptron (MLP)

This folder contains a notebook demonstration of a **multilayer perceptron (MLP)**: a feedforward neural network trained with backpropagation.

The notebook emphasizes the forward pass, loss computation, gradient flow, and how training updates parameters over time.

---

## Design Philosophy

- **From-scratch learning**: show the core mechanics without black-box training loops
- **Transparency**: forward + backward computations are explicit
- **Educational defaults**: small networks and datasets so behavior can be inspected

---

## Contents

```text
Multilayer Perceptron/
├── multilayer_perceptron_example.ipynb
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

Open `multilayer_perceptron_example.ipynb` and run all cells.

---

## What This Demonstrates

- How stacking layers enables non-linear decision boundaries
- The role of activations in expressiveness
- How backpropagation computes gradients efficiently

---

## Limitations

- Simplified educational implementation (not optimized, limited safeguards)
- Not intended for large-scale deep learning workloads

---

## Conclusion

This notebook provides a transparent MLP walkthrough to connect neural network math to executable code.
