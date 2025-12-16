# k-Nearest Neighbors (kNN) (Notebook)

This notebook demonstrates **kNN classification** on a labeled dataset, emphasizing how predictions are driven by the choice of \(k\), the distance metric, and feature scaling.

---

## Design Philosophy

- **Distance-based intuition**: make “nearest” concrete
- **Transparent predictions**: connect neighbors to labels
- **Parameter exploration**: see how \(k\) and scaling change outcomes

---

## Files

- **Notebook**: `KNN_example.ipynb`
- **Dataset**: `iris.csv`

---

## What this notebook covers

- kNN decision logic for classification  
- Sensitivity to \(k\), tie-breaking, and distance metrics  
- Why normalization/standardization matters for distance-based models

---

## How to run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open `KNN_example.ipynb` and run all cells. The notebook expects `iris.csv` to be in the same folder.

---

## Limitations

- kNN can be slow at prediction time for large datasets
- Highly sensitive to feature scaling and irrelevant dimensions

