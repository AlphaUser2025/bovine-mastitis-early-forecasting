"""
data_prep.py  --  Module 1 of the ANFIS bovine-mastitis project
================================================================
What this script does (in order):
  1. Load + validate the raw CSV, normalise column names / the 'hostlene' typo
  2. LEAKAGE AUDIT  -> proves Milk_visibility == class1, drops it, saves evidence
  3. WITHIN-COW AUDIT -> does a cow's label change across Day 1..6? (decides if
     "forecasting" is even possible with this data; see report note)
  4. Cow-GROUPED + stratified split (StratifiedGroupKFold) -> no Cow_ID leakage
  5. Feature engineering: Udder Swelling Index + Udder Asymmetry Index
  6. Min-max scaling FITTED ON TRAIN ONLY (no test information leaks)
  7. Save processed data, feature-set definitions, scaler params, audit JSON
  8. Save summary plots for the Review II slides

Run from the project root:
    python data_prep.py                      # default paths
    python data_prep.py --input path/to.csv  # custom raw file

Expected layout (created automatically where missing):
    data/raw/clinical_mastitis_cows.csv      <- you put the file here
    data/processed/                          <- outputs (CSV + JSON)
    reports/figures/                         <- PNG plots
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; plots are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------- #
# Configuration (single source of truth; import these from other modules)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_FOLDS = 5          # 5 folds -> each fold ~20% of cows
TEST_FOLD = 0        # fold 0 is the held-out test set; folds 1-4 = train
                     # (the `fold` column is also reusable for grouped 5-fold CV later)

# Quadrant -> (inner sensor, external sensor). F/R = front/rear, L/R = left/right.
# (Assumption: I* and E* are two sensor channels per quadrant; confirm exact
#  meaning against the Mendeley data description before writing the paper.)
QUADRANTS = {
    "FL": ("IUFL", "EUFL"),
    "FR": ("IUFR", "EUFR"),
    "RL": ("IURL", "EURL"),
    "RR": ("IURR", "EURR"),
}
SENSOR_COLS = [c for pair in QUADRANTS.values() for c in pair]

RAW_RENAME = {
    "Cow_ID": "cow_id",
    "Day": "day",
    "Breed": "breed",
    "Months after giving birth": "months_pp",
    "Previous_Mastits_status": "prev_mastitis",  # (sic) spelling in source file
    "Temperature": "temperature",
    "Hardness": "hardness",
    "Pain": "pain",
    "Milk_visibility": "milk_visibility",
    "class1": "label",
}
EXPECTED_RAW_COLS = ["Cow_ID", "Day", "Breed", "Months after giving birth",
                     "Previous_Mastits_status", *SENSOR_COLS, "Temperature",
                     "Hardness", "Pain", "Milk_visibility", "class1"]

# Columns that get min-max scaled -> name of scaled column
SCALE_MAP = {
    "swelling_index_raw": "swelling_index",
    "asymmetry_index_raw": "asymmetry_index",
    "temperature": "temperature_n",
    "months_pp": "months_pp_n",
}

# The two experimental feature sets (the "impress the HAG" comparison)
EARLY_FEATURES = ["swelling_index", "asymmetry_index", "temperature_n",
                  "months_pp_n", "breed_enc"]                  # no human-observed signs
FULL_FEATURES = EARLY_FEATURES + ["hardness", "pain"]          # + palpation-type signs

log = logging.getLogger("data_prep")


# --------------------------------------------------------------------------- #
# Step 1 - load & validate
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in EXPECTED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV is missing expected columns: {missing}")
    if df[EXPECTED_RAW_COLS].isna().any().any():
        raise ValueError("Unexpected missing values in raw CSV")
    df = df.rename(columns=RAW_RENAME)

    # Source file spells Holstein as 'hostlene' -> normalise, keep Jersey
    df["breed"] = (df["breed"].astype(str).str.strip().str.lower()
                   .map({"jersey": "Jersey", "hostlene": "Holstein",
                         "holstein": "Holstein"}))
    if df["breed"].isna().any():
        raise ValueError("Unrecognised Breed value(s) found")
    df["breed_enc"] = (df["breed"] == "Holstein").astype(int)  # Jersey=0, Holstein=1
    log.info("Loaded %s rows x %s cols", *df.shape)
    return df


# --------------------------------------------------------------------------- #
# Step 2 - leakage audit
# --------------------------------------------------------------------------- #
def audit_leakage(df: pd.DataFrame) -> dict:
    """Proves/quantifies target leakage. Evidence goes straight into the report."""
    identical = bool((df["milk_visibility"] == df["label"]).all())
    corr = float(np.corrcoef(df["milk_visibility"], df["label"])[0, 1])
    crosstab = pd.crosstab(df["milk_visibility"], df["label"])

    # "Single-rule accuracy": how well does ONE binary column alone predict the label?
    stump_acc = {}
    for col in ["milk_visibility", "hardness", "pain", "prev_mastitis"]:
        agree = float((df[col] == df["label"]).mean())
        stump_acc[col] = round(max(agree, 1 - agree), 4)

    audit = {
        "milk_visibility_identical_to_label": identical,
        "pearson_milk_visibility_vs_label": round(corr, 6),
        "crosstab_milk_visibility_vs_label": {
            f"mv={i}": {f"label={j}": int(crosstab.loc[i, j]) for j in crosstab.columns}
            for i in crosstab.index},
        "single_binary_feature_accuracy": stump_acc,
        "action": "milk_visibility DROPPED from all feature sets",
    }
    log.warning("LEAKAGE: milk_visibility identical to label = %s (r=%.4f)", identical, corr)
    log.info("Single-binary-feature accuracy: %s", stump_acc)
    return audit


# --------------------------------------------------------------------------- #
# Step 3 - within-cow audit
# --------------------------------------------------------------------------- #
def audit_cows(df: pd.DataFrame) -> dict:
    g = df.groupby("cow_id")
    n_unique_labels = g["label"].nunique()
    audit = {
        "n_cows": int(g.ngroups),
        "rows_per_cow_distribution": {str(k): int(v) for k, v in g.size().value_counts().items()},
        "cows_with_constant_label": int((n_unique_labels == 1).sum()),
        "cows_whose_label_changes_across_days": int((n_unique_labels > 1).sum()),
        "max_breeds_per_cow": int(g["breed"].nunique().max()),
    }
    audit["label_constant_per_cow"] = audit["cows_whose_label_changes_across_days"] == 0
    if audit["label_constant_per_cow"]:
        log.warning("Label is CONSTANT within every cow -> Day 1-6 are repeated "
                    "measurements of one status, not an onset trajectory. Frame the "
                    "task as early DETECTION without clinical signs, not time-ahead "
                    "forecasting, unless you can justify otherwise.")
    else:
        log.info("%d cows change label across days -> temporal/lag features are possible.",
                 audit["cows_whose_label_changes_across_days"])
    return audit


# --------------------------------------------------------------------------- #
# Step 4 - cow-grouped, stratified split
# --------------------------------------------------------------------------- #
def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Why grouped: each cow has up to 6 near-identical rows. A plain random split puts
    Day 2 of cow17 in train and Day 5 of cow17 in test -> the model memorises the
    cow's sensor 'fingerprint' and test accuracy is inflated.
    Why also stratified: StratifiedGroupKFold keeps ~the same healthy/mastitis ratio
    in every fold while never splitting a cow across folds.
    """
    df = df.reset_index(drop=True).copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for k, (_, test_idx) in enumerate(sgkf.split(df, df["label"], groups=df["cow_id"])):
        df.loc[test_idx, "fold"] = k
    assert (df["fold"] >= 0).all(), "Some rows were not assigned to a fold"

    # Hard guarantee: no cow appears in both train and test
    test_cows = set(df.loc[df["fold"] == TEST_FOLD, "cow_id"])
    train_cows = set(df.loc[df["fold"] != TEST_FOLD, "cow_id"])
    assert test_cows.isdisjoint(train_cows), "Cow leakage between train and test!"
    return df


# --------------------------------------------------------------------------- #
# Step 5 - feature engineering
# --------------------------------------------------------------------------- #
def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Udder Swelling Index  = mean of all 8 sensor readings  (overall 'size/fullness')
    Udder Asymmetry Index = (max - min) / mean of the 4 quadrant totals
        quadrant total Q_k = inner_k + external_k
    Intuition: mastitis usually hits ONE quarter. That quarter swells, the others
    don't, so the spread between the biggest and smallest quarter (relative to the
    udder average) jumps. Dividing by the mean makes it scale-free across cows.
    Both indices use only the current row -> no train/test leakage possible.
    """
    df = df.copy()
    q = pd.DataFrame({k: df[i] + df[e] for k, (i, e) in QUADRANTS.items()})
    df["swelling_index_raw"] = df[SENSOR_COLS].mean(axis=1)
    df["asymmetry_index_raw"] = (q.max(axis=1) - q.min(axis=1)) / q.mean(axis=1)
    return df


# --------------------------------------------------------------------------- #
# Step 6 - scaling (fit on train only)
# --------------------------------------------------------------------------- #
def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    raw_cols = list(SCALE_MAP.keys())
    train_mask = df["fold"] != TEST_FOLD

    scaler = MinMaxScaler()
    scaler.fit(df.loc[train_mask, raw_cols])            # <- train rows ONLY
    scaled = scaler.transform(df[raw_cols])

    n_clipped = 0
    for j, (raw, new) in enumerate(SCALE_MAP.items()):
        col = scaled[:, j]
        outside = ((col < -1e-9) | (col > 1 + 1e-9)) & ~train_mask.values  # test rows only
        n_clipped += int(outside.sum())
        df[new] = np.clip(col, 0.0, 1.0)                # unseen test extremes -> [0,1]

    params = {
        "columns": raw_cols,
        "scaled_names": list(SCALE_MAP.values()),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "fitted_on": "train rows only (fold != TEST_FOLD)",
        "n_test_values_clipped_to_unit_interval": n_clipped,
    }
    log.info("Scaler fitted on train; %d test values fell outside train range and were clipped", n_clipped)
    return df, params


# --------------------------------------------------------------------------- #
# Step 8 - plots
# --------------------------------------------------------------------------- #
CLASS_NAMES = {0: "Healthy", 1: "Mastitis"}
CLASS_COLORS = {0: "#2a9d8f", 1: "#e76f51"}


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def make_plots(df: pd.DataFrame, leak: dict, fig_dir: Path) -> None:
    train = df[df["fold"] != TEST_FOLD]
    test = df[df["fold"] == TEST_FOLD]

    # Fig 1: class balance
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = df["label"].value_counts().sort_index()
    ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
           color=[CLASS_COLORS[i] for i in counts.index])
    for x, v in enumerate(counts.values):
        ax.text(x, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Rows"); ax.set_title("Class distribution (class1)")
    ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, fig_dir / "fig1_class_distribution.png")

    # Fig 2: leakage evidence (Milk_visibility vs label)
    ct = leak["crosstab_milk_visibility_vs_label"]
    mat = np.array([[ct[f"mv={i}"][f"label={j}"] for j in (0, 1)] for i in (0, 1)])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.imshow(mat, cmap="Reds")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Healthy", "Mastitis"]); ax.set_yticks([0, 1], ["mv = 0", "mv = 1"])
    ax.set_xlabel("class1 (label)"); ax.set_ylabel("Milk_visibility")
    ax.set_title("Target leakage: Milk_visibility == class1\n(off-diagonals are zero -> dropped)",
                 fontsize=10)
    _save(fig, fig_dir / "fig2_leakage_milk_visibility.png")

    # Fig 3: correlation of every candidate feature with the label (train only)
    cand = SENSOR_COLS + ["swelling_index_raw", "asymmetry_index_raw", "temperature",
                          "months_pp", "breed_enc", "prev_mastitis", "hardness", "pain"]
    corr = train[cand].corrwith(train["label"]).sort_values()
    colors = ["#264653" if c in ("swelling_index_raw", "asymmetry_index_raw")
              else "#e9c46a" if c in ("hardness", "pain") else "#8d99ae" for c in corr.index]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_xlabel("Pearson r with label (train set)")
    ax.set_title("Feature-label correlation\n(dark = engineered, yellow = late clinical signs)",
                 fontsize=10)
    _save(fig, fig_dir / "fig3_feature_label_correlation.png")

    # Fig 4: engineered index distributions by class
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, title in zip(axes, ["swelling_index_raw", "asymmetry_index_raw"],
                              ["Udder Swelling Index", "Udder Asymmetry Index"]):
        for c in (0, 1):
            ax.hist(train.loc[train["label"] == c, col], bins=40, alpha=0.55, density=True,
                    color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        ax.set_title(title); ax.set_xlabel("value (unscaled)"); ax.legend()
    axes[0].set_ylabel("Density")
    _save(fig, fig_dir / "fig4_index_distributions.png")

    # Fig 5: SI vs AI scatter
    samp = train.sample(min(2500, len(train)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for c in (0, 1):
        s = samp[samp["label"] == c]
        ax.scatter(s["swelling_index_raw"], s["asymmetry_index_raw"], s=7, alpha=0.4,
                   color=CLASS_COLORS[c], label=CLASS_NAMES[c])
    ax.set_xlabel("Swelling Index"); ax.set_ylabel("Asymmetry Index")
    ax.set_title("2-D feature space the ANFIS will partition"); ax.legend(markerscale=2)
    _save(fig, fig_dir / "fig5_swelling_vs_asymmetry.png")

    # Fig 6: split sanity (rows + class ratio)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].bar(["Train", "Test"], [len(train), len(test)], color=["#457b9d", "#f4a261"])
    axes[0].set_title("Rows per split")
    axes[1].bar(["Train", "Test"], [train["label"].mean(), test["label"].mean()],
                color=["#457b9d", "#f4a261"])
    axes[1].set_ylim(0, 1); axes[1].set_title("Mastitis fraction per split")
    for ax, vals in zip(axes, [[len(train), len(test)],
                               [train["label"].mean(), test["label"].mean()]]):
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.3g}" if v < 1 else f"{int(v)}", ha="center", va="bottom")
    fig.suptitle(f"Cow-grouped split: {train['cow_id'].nunique()} train cows / "
                 f"{test['cow_id'].nunique()} test cows, zero overlap", fontsize=10)
    _save(fig, fig_dir / "fig6_split_sanity.png")

    # Fig 7: correlation heatmap of the model features
    cols = FULL_FEATURES + ["label"]
    cm = train[cols].corr().values
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046); ax.set_title("Correlation of model features (train)")
    _save(fig, fig_dir / "fig7_feature_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Module 1: data preparation")
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--input", default=None, help="raw CSV (default data/raw/clinical_mastitis_cows.csv)")
    args = ap.parse_args()

    root = Path(args.root)
    raw_path = Path(args.input) if args.input else root / "data/raw/clinical_mastitis_cows.csv"
    proc_dir, fig_dir = root / "data/processed", root / "reports/figures"
    proc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = load_raw(raw_path)

    leakage = audit_leakage(df)
    cow_audit = audit_cows(df)
    df = df.drop(columns=["milk_visibility"])            # <- the leakage fix

    df = assign_folds(df)                                 # split BEFORE scaling
    df = add_indices(df)
    df, scaler_params = scale_features(df)

    id_cols = ["cow_id", "day", "label", "fold", "breed"]
    keep = (id_cols + SENSOR_COLS + ["prev_mastitis", "hardness", "pain", "breed_enc"]
            + list(SCALE_MAP.keys()) + list(SCALE_MAP.values()))
    out = df[keep]
    train, test = out[out["fold"] != TEST_FOLD], out[out["fold"] == TEST_FOLD]

    out.to_csv(proc_dir / "full_processed.csv", index=False)
    train.to_csv(proc_dir / "train.csv", index=False)
    test.to_csv(proc_dir / "test.csv", index=False)

    summary = {
        "raw_shape": list(pd.read_csv(raw_path).shape),
        "leakage_audit": leakage,
        "cow_audit": cow_audit,
        "split": {
            "method": f"StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, "
                      f"random_state={RANDOM_STATE}); fold {TEST_FOLD} = test",
            "train_rows": len(train), "test_rows": len(test),
            "train_cows": int(train["cow_id"].nunique()),
            "test_cows": int(test["cow_id"].nunique()),
            "train_mastitis_fraction": round(float(train["label"].mean()), 4),
            "test_mastitis_fraction": round(float(test["label"].mean()), 4),
            "cow_overlap": 0,
        },
        "scaler": scaler_params,
        "feature_sets": {"early_warning": EARLY_FEATURES, "full_clinical": FULL_FEATURES},
        "notes": [
            "prev_mastitis kept in CSVs but excluded from feature sets: ~0.4% positive rows "
            "(near-zero variance) -> a Gaussian membership function cannot learn from it.",
            "breed_enc: Jersey=0, Holstein=1 (source spelling 'hostlene' normalised).",
        ],
    }
    (proc_dir / "data_prep_summary.json").write_text(json.dumps(summary, indent=2))

    make_plots(out, leakage, fig_dir)

    log.info("DONE. train=%d rows (%d cows) | test=%d rows (%d cows)",
             len(train), train["cow_id"].nunique(), len(test), test["cow_id"].nunique())
    log.info("Outputs -> %s and %s", proc_dir, fig_dir)


if __name__ == "__main__":
    main()
