# Ensemble Methods (Notebook)

This notebook demonstrates core **ensemble learning** ideas: training multiple models and combining them to improve stability and/or accuracy relative to a single learner.

---

## Design Philosophy

- **Concept-first**: show why ensembling helps (variance reduction, robustness)
- **Transparent evaluation**: compare single vs. combined models
- **Experimentation encouraged**: vary the number/diversity of learners

---

## Files

- **Notebook**: `ensemble_methods_example.ipynb`

---

## What this notebook covers

- Motivations for ensembles and when they help  
- Combining predictions (e.g., voting/averaging)  
- Empirical comparison against individual learners

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `ensemble_methods_example.ipynb` and run all cells.

---

## Limitations

- Focused on educational mechanics, not production-grade ensembling
- Results depend on base learner diversity and data regime

