"""
baselines.py  --  Module 2: reference models the ANFIS must be compared against
================================================================================
Trains on data/processed/train.csv, scores on data/processed/test.csv, for BOTH
feature sets (Early-Warning vs Full-Clinical) defined in data_prep.py.

Models (each one answers a different "why not just use X?" question):
  Majority class   -> the floor. Anything below this is broken.
  Logistic Reg.    -> the simplest linear, explainable model.
  Decision Tree d=4-> the main INTERPRETABILITY rival (small if-then rules).
  Random Forest    -> the black-box accuracy ceiling the ANFIS is judged against.

Run:  python baselines.py
Out:  reports/baseline_results.csv  and  reports/figures/fig8_baseline_roc.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier

from data_prep import EARLY_FEATURES, FULL_FEATURES, RANDOM_STATE
from metrics import evaluate

log = logging.getLogger("baselines")

FEATURE_SETS = {"Early-Warning": EARLY_FEATURES, "Full-Clinical": FULL_FEATURES}


def make_models() -> dict:
    return {
        "Majority class": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree (d=4)": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                                n_jobs=-1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    train = pd.read_csv(root / "data/processed/train.csv")
    test = pd.read_csv(root / "data/processed/test.csv")
    (root / "reports/figures").mkdir(parents=True, exist_ok=True)

    rows, roc_data = [], {}
    for fs_name, feats in FEATURE_SETS.items():
        Xtr, ytr = train[feats].values, train["label"].values
        Xte, yte = test[feats].values, test["label"].values
        for m_name, model in make_models().items():
            model.fit(Xtr, ytr)
            prob = model.predict_proba(Xte)[:, 1]
            res = evaluate(yte, prob, test["cow_id"].values)
            rows.append({"feature_set": fs_name, "model": m_name, **res})
            roc_data[(fs_name, m_name)] = (yte, prob)
            log.info("%-13s | %-20s | row F1=%.3f AUC=%.3f | cow F1=%.3f",
                     fs_name, m_name, res["row_f1"], res["row_auc"], res["cow_f1"])

    df = pd.DataFrame(rows)
    df.to_csv(root / "reports/baseline_results.csv", index=False)

    # ROC curves, one panel per feature set
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, fs_name in zip(axes, FEATURE_SETS):
        for m_name in make_models():
            if m_name == "Majority class":
                continue
            y, p = roc_data[(fs_name, m_name)]
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=m_name)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{fs_name} features"); ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("True positive rate")
    fig.savefig(root / "reports/figures/fig8_baseline_roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved reports/baseline_results.csv and fig8_baseline_roc.png")


if __name__ == "__main__":
    main()
