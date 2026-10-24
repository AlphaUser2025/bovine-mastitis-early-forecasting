"""
explain.py  --  Layer 4 of the architecture: explainable output
================================================================
  describe_rules : the whole model as readable IF-THEN rules
  explain_row    : for ONE animal -> score, risk tier, and the rules that fired
  fuzzy_risk     : score -> memberships in Low / Watch / High (the "fuzzy" tier)
  rank_cows      : cow-level risk ranking (mean of the 6 daily scores)

This is the project's core contribution: a black-box (Random Forest) can give a
score, but only the ANFIS can say WHICH rules fired and how strongly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BINARY = {"breed_enc", "hardness", "pain"}


def _label(feature: str, value: float) -> str:
    if feature in BINARY:
        return "YES" if value >= 0.5 else "NO"
    return "LOW" if value < 0.33 else ("MEDIUM" if value < 0.66 else "HIGH")


def _rule_text(model, i: int, names: list[str]) -> str:
    cond = " AND ".join(f"{n} is {_label(n, model.c[i, j])}" for j, n in enumerate(names))
    proto = float(np.append(model.c[i], 1.0) @ model.theta[i])  # rule output at its prototype
    return f"IF {cond} THEN risk ~ {proto:.2f}"


def describe_rules(model, names: list[str]) -> str:
    lines = [f"Rule {i + 1}: {_rule_text(model, i, names)}" for i in range(model.R)]
    return "\n".join(lines)


def fuzzy_risk(score: float) -> dict:
    """Trapezoid/triangle membership of a risk score in Low / Watch / High."""
    s = float(np.clip(score, 0.0, 1.0))
    low = 1.0 if s <= 0.3 else max(0.0, (0.5 - s) / 0.2)
    high = 1.0 if s >= 0.7 else max(0.0, (s - 0.5) / 0.2)
    watch = max(0.0, 1.0 - abs(s - 0.5) / 0.2)
    return {"Low": round(low, 3), "Watch": round(watch, 3), "High": round(high, 3)}


def risk_tier(score: float) -> str:
    m = fuzzy_risk(score)
    return max(m, key=m.get)


def explain_row(model, x: np.ndarray, names: list[str], top: int = 3) -> dict:
    x = np.asarray(x, float)[None, :]
    wbar = model._wbar(x)[0]
    f = model._rule_outputs(x)[0]
    score = float((wbar * f).sum())
    order = np.argsort(-wbar)[:top]
    return {
        "score": round(score, 3),
        "tier": risk_tier(score),
        "tier_memberships": fuzzy_risk(score),
        "top_rules": [{"rule": int(i) + 1, "firing_share": round(float(wbar[i]), 3),
                       "rule_output": round(float(f[i]), 3), "text": _rule_text(model, i, names)}
                      for i in order],
    }


def rank_cows(cow_ids, scores) -> pd.DataFrame:
    d = pd.DataFrame({"cow_id": cow_ids, "score": scores}).groupby("cow_id", as_index=False).mean()
    d["tier"] = d["score"].map(risk_tier)
    return d.sort_values("score", ascending=False).reset_index(drop=True)
