# Interpretable ANFIS for Early Bovine Mastitis Detection

Soft Computing project (M.Tech, VIT-AP). Problem statement SIH26109.
Idea: detect mastitis from udder sensor readings with a **rule-based, explainable** ANFIS and compare it with black-box ML.

## Quick start (new collaborator: 15 minutes)

```bash
git clone <repo-url> && cd mastitis-anfis
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# get clinical_mastitis_cows.csv from your teammate (or the Mendeley page) and put it in data/raw/
python test_modules.py      # 4 x OK expected
python data_prep.py         # Module 1  -> data/processed/, reports/figures/fig1-7
python baselines.py         # Module 2  -> reports/baseline_results.csv
python train_anfis.py       # Modules 3-5 -> reports/anfis_results.csv, comparison.csv, rules, fig9-10
```

Same random seeds everywhere, so **you should get exactly the same numbers as your teammate.** If not, tell them: that is a bug worth finding.

## Pipeline

```
raw CSV -> data_prep.py (drop leakage, cow-grouped split, 2 indices, train-only scaling)
        -> baselines.py (LogReg / Tree / RandomForest)            \
        -> subtractive_clustering.py (rules from data) -> anfis.py (hybrid learning) -> explain.py
                                                                   /  compared in reports/comparison.csv
```

## Where to read next

1. `docs/HANDOFF.md` - every design decision, its reason, and open issues (read before touching code).
2. `TASKS.md` - who is doing what, with dates.
3. `JOURNAL.md` - research journal; add an entry every session.

## Team workflow

- `main` always runs. Never commit straight to `main`; use a branch `feat/<what>-<name>` and a pull request.
- The OTHER person reviews and merges. Reviewing means you can explain the change in 5 minutes.
- Raw data is not committed (`.gitignore`); share it by Drive/WhatsApp.
- Numbers on slides or in the paper must come from files in `reports/`. No invented numbers, no invented citations.
