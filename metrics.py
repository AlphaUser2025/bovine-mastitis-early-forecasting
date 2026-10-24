"""
metrics.py  --  shared evaluation utilities (used by baselines AND the ANFIS)
==============================================================================
Every model in the project is scored through `evaluate()` so all numbers are
comparable. Three ideas live here:

1. ROW-level metrics   : every (cow, day) row is one prediction.
2. COW-level metrics   : the 6 daily scores of a cow are averaged into ONE
                         prediction per cow. This is the honest unit, because
                         the label is (almost) constant within a cow.
3. COW-BOOTSTRAP CI    : resample whole COWS (not rows) with replacement to get
                         confidence intervals. Rows of one cow are not independent,
                         so a row-level bootstrap would give falsely tight intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def _core(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),      # sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def cow_level(y_true, y_prob, cow_ids) -> tuple[np.ndarray, np.ndarray]:
    """Average the daily scores per cow; cow label = majority of its daily labels."""
    d = pd.DataFrame({"y": y_true, "p": y_prob, "c": cow_ids})
    g = d.groupby("c").agg(y=("y", "mean"), p=("p", "mean"))
    return (g["y"].values >= 0.5).astype(int), g["p"].values


def evaluate(y_true, y_prob, cow_ids, thr: float = 0.5,
             n_boot: int = 1000, seed: int = 42) -> dict:
    """Return row-level metrics, cow-level metrics, and 95% cow-bootstrap CIs."""
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    cow_ids = np.asarray(cow_ids)

    out = {f"row_{k}": v for k, v in _core(y_true, y_prob, thr).items()}
    yc, pc = cow_level(y_true, y_prob, cow_ids)
    out.update({f"cow_{k}": v for k, v in _core(yc, pc, thr).items()})

    # cow bootstrap on the ROW-level F1 and AUC (resample cows, keep all their rows)
    rng = np.random.default_rng(seed)
    cows = np.unique(cow_ids)
    idx_of = {c: np.where(cow_ids == c)[0] for c in cows}
    f1s, aucs = [], []
    for _ in range(n_boot):
        pick = rng.choice(cows, size=len(cows), replace=True)
        idx = np.concatenate([idx_of[c] for c in pick])
        m = _core(y_true[idx], y_prob[idx], thr)
        f1s.append(m["f1"]); aucs.append(m["auc"])
    if n_boot > 0:
        for name, vals in (("f1", f1s), ("auc", aucs)):
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            out[f"row_{name}_ci95"] = (round(float(lo), 4), round(float(hi), 4))
    return out
