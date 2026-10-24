"""
audit_odd_cow.py - find cows whose label changes across Day 1-6.
Why it matters: if the label is constant within a cow, the data supports early DETECTION,
not time-ahead FORECASTING. The one exception cow may be a data-entry error.

Run:  python audit_odd_cow.py [path/to/clinical_mastitis_cows.csv]
"""
import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/clinical_mastitis_cows.csv"
df = pd.read_csv(path)
n_labels = df.groupby("Cow_ID")["class1"].nunique()
odd = n_labels[n_labels > 1].index.tolist()
print(f"{len(odd)} cow(s) whose class1 changes across days: {odd}")
cols = ["Cow_ID", "Day", "Breed", "Temperature", "Hardness", "Pain", "class1"]
for cow in odd:
    print(f"\n--- {cow} ---")
    print(df.loc[df["Cow_ID"] == cow, cols].to_string(index=False))
