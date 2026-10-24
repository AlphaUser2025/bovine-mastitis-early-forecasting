"""
train_anfis.py  --  Module 5: train + evaluate the ANFIS on both feature sets
==============================================================================
Protocol (no peeking at the test fold):
  1. Inside the TRAIN folds, hold out one fold (VAL_FOLD, whole cows) as validation.
  2. For each cluster radius ra in RA_GRID: cluster -> build rules -> hybrid-train with
     early stopping on validation. Pick the ra with the best VALIDATION AUC.
  3. Refit on ALL training cows with the chosen ra and epoch count.
  4. Score ONCE on the test fold (the 220 unseen cows). Report row + cow level.
  Radius ra controls the number of rules: small ra -> many specific rules,
  large ra -> few general rules. (GA tuning of ra, rb is the Final Review extension.)

Run:   python train_anfis.py
Out:   reports/anfis_results.csv, anfis_ra_grid.csv, comparison.csv, rules_*.txt,
       example_explanations.txt, models/anfis_*.npz, figures fig9 + fig10_*
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anfis import ANFIS
from data_prep import EARLY_FEATURES, FULL_FEATURES
from explain import describe_rules, explain_row, rank_cows
from metrics import evaluate
from subtractive_clustering import initial_rule_params

log = logging.getLogger("train_anfis")

FEATURE_SETS = {"Early-Warning": EARLY_FEATURES, "Full-Clinical": FULL_FEATURES}
RA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
MAX_RULES = 12
VAL_FOLD = 1
EPOCHS, LR = 100, 0.01


def build(Xtr, ytr, ra):
    C, S = initial_rule_params(Xtr, ytr, ra=ra, use_target=True, max_rules=MAX_RULES)
    return ANFIS(C, S)


def plot_curve(histories: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, h in histories.items():
        ax.plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label=f"{name} train")
        if "val_loss" in h[0]:
            ax.plot([r["epoch"] for r in h], [r["val_loss"] for r in h], "--", label=f"{name} val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss  0.5*MSE"); ax.legend(fontsize=8)
    ax.set_title("ANFIS hybrid-learning curves")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_mfs(model, names, title, path: Path):
    x = np.linspace(-0.1, 1.1, 300)
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 2.8), sharey=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        for i in range(model.R):
            ax.plot(x, np.exp(-0.5 * ((x - model.c[i, j]) / model.s[i, j]) ** 2), lw=1)
        ax.set_title(names[j], fontsize=9); ax.set_xlabel("scaled value")
    np.atleast_1d(axes)[0].set_ylabel("membership")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    (root / "reports/figures").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)

    train = pd.read_csv(root / "data/processed/train.csv")
    test = pd.read_csv(root / "data/processed/test.csv")
    inner_tr, inner_va = train[train["fold"] != VAL_FOLD], train[train["fold"] == VAL_FOLD]

    grid_rows, final_rows, curves, examples = [], [], {}, []
    for fs, feats in FEATURE_SETS.items():
        Xtr, ytr = inner_tr[feats].values, inner_tr["label"].values
        Xva, yva = inner_va[feats].values, inner_va["label"].values

        # ---- step 2: choose ra on validation ---- #
        best = None
        for ra in RA_GRID:
            m = build(Xtr, ytr, ra).fit(Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=LR)
            v = evaluate(yva, m.predict_score(Xva), inner_va["cow_id"].values, n_boot=0)
            grid_rows.append({"feature_set": fs, "ra": ra, "n_rules": m.R,
                              "best_epoch": m.best_epoch_, "val_auc": v["row_auc"],
                              "val_f1": v["row_f1"]})
            log.info("%-13s ra=%.1f rules=%2d epoch=%3d | val AUC=%.3f F1=%.3f",
                     fs, ra, m.R, m.best_epoch_, v["row_auc"], v["row_f1"])
            if best is None or v["row_auc"] > best[0]:
                best = (v["row_auc"], ra, m.best_epoch_)
        _, ra, n_ep = best

        # ---- step 3: refit on all training cows ---- #
        Xall, yall = train[feats].values, train["label"].values
        model = build(Xall, yall, ra).fit(Xall, yall, epochs=max(n_ep, 5), lr=LR)
        curves[fs] = model.history

        # ---- step 4: one-shot test evaluation ---- #
        Xte, yte = test[feats].values, test["label"].values
        score = model.predict_score(Xte)
        res = evaluate(yte, score, test["cow_id"].values)
        final_rows.append({"feature_set": fs, "model": "ANFIS (subtractive, hybrid)", "ra": ra,
                           "n_rules": model.R, "epochs": max(n_ep, 5), **res})
        log.info("%-13s TEST | rules=%d | row F1=%.3f AUC=%.3f | cow F1=%.3f",
                 fs, model.R, res["row_f1"], res["row_auc"], res["cow_f1"])

        # ---- explainability artefacts ---- #
        (root / f"reports/rules_{fs}.txt").write_text(describe_rules(model, feats))
        model.save(root / f"models/anfis_{fs}.npz")
        plot_mfs(model, feats, f"Learned membership functions - {fs} ({model.R} rules, ra={ra})",
                 root / f"reports/figures/fig10_membership_{fs}.png")
        hi, lo = int(np.argmax(score)), int(np.argmin(score))
        for tag, i in (("highest-risk test row", hi), ("lowest-risk test row", lo)):
            e = explain_row(model, Xte[i], feats)
            examples.append(f"[{fs}] {tag} (cow {test['cow_id'].iloc[i]}, true label "
                            f"{int(yte[i])})\n{json.dumps(e, indent=2)}\n")
        rank_cows(test["cow_id"].values, score).head(10).to_csv(
            root / f"reports/top10_risk_cows_{fs}.csv", index=False)

    plot_curve({k: v for k, v in curves.items()}, root / "reports/figures/fig9_anfis_learning_curve.png")
    (root / "reports/example_explanations.txt").write_text("\n".join(examples))

    pd.DataFrame(grid_rows).to_csv(root / "reports/anfis_ra_grid.csv", index=False)
    anfis_df = pd.DataFrame(final_rows)
    anfis_df.to_csv(root / "reports/anfis_results.csv", index=False)

    base_path = root / "reports/baseline_results.csv"
    if base_path.exists():
        cols = ["feature_set", "model", "row_accuracy", "row_precision", "row_recall",
                "row_specificity", "row_f1", "row_auc", "cow_f1", "cow_auc"]
        both = pd.concat([pd.read_csv(base_path), anfis_df], ignore_index=True)
        both[[c for c in cols if c in both.columns]].round(4).to_csv(
            root / "reports/comparison.csv", index=False)
        log.info("Wrote reports/comparison.csv (baselines + ANFIS)")
    log.info("DONE")


if __name__ == "__main__":
    main()
