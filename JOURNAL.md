# Research journal

Add an entry every session: date, who, what you did, what you learned, what is next. Newest at the bottom. This is viva evidence.

## 2026-10-07 - A
- Ran `data_prep.py` on the real CSV (6600 rows x 18 cols, 1100 cows x 6 days).
- Found: `Milk_visibility` is identical to `class1` (r = 1.0), so it is the target; dropped.
- Cow-grouped split: 880 train cows / 220 test cows, zero overlap; mastitis fraction 0.3941 train / 0.4227 test.
- Label is constant within 1099 of 1100 cows. One cow changes. Single-rule accuracy: Hardness 0.8718, Pain 0.7729.
- Next: baselines.

## 2026-10-14 - A
- Received `baselines.py` and `metrics.py` (row-level and cow-level metrics, cow-bootstrap CIs). Results not yet recorded.

## 2026-10-22 - A
- Received Modules 3-5 (`subtractive_clustering.py`, `anfis.py`, `explain.py`, `train_anfis.py`, `test_modules.py`); unit tests and a synthetic-data dry run passed. Not yet run on real data.

## 2026-10-24 - A
- Set up the shared repo with B. Review II is on Tuesday 2026-10-27.
- Next: run all scripts on both machines and compare numbers.
