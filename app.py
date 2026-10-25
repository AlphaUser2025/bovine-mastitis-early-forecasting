from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from anfis import ANFIS
from data_prep import EARLY_FEATURES, FULL_FEATURES, QUADRANTS
from explain import explain_row, risk_tier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Bovine Mastitis Detection",
    page_icon="🐄",
    layout="wide"
)

@st.cache_resource
def load_models():
    return {
        "Early-Warning": ANFIS.load(MODEL_DIR / "anfis_Early-Warning.npz"),
        "Full-Clinical": ANFIS.load(MODEL_DIR / "anfis_Full-Clinical.npz"),
    }

models = load_models()

st.title("🐄 Bovine Mastitis Detection")
st.write(
    "ANFIS-based research prototype for estimating mastitis risk "
    "from udder sensor readings and clinical observations."
)

model_name = st.selectbox(
    "Choose model",
    ["Early-Warning", "Full-Clinical"]
)

st.caption(
    "Early-Warning uses sensor-derived features. "
    "Full-Clinical additionally uses hardness and pain."
)

with st.form("prediction_form"):
    st.subheader("Cow information")

    c1, c2, c3 = st.columns(3)
    with c1:
        breed = st.selectbox("Breed", ["Jersey", "Holstein"])
        months_pp = st.number_input(
            "Months after giving birth", min_value=0.0,
            max_value=24.0, value=3.0, step=1.0
        )
    with c2:
        temperature = st.number_input(
            "Temperature", min_value=0.0, max_value=100.0,
            value=38.5, step=0.1
        )
    with c3:
        hardness = st.number_input(
            "Hardness (dataset scale)", min_value=0.0,
            max_value=1.0, value=0.0, step=0.1
        )
        pain = st.number_input(
            "Pain (dataset scale)", min_value=0.0,
            max_value=1.0, value=0.0, step=0.1
        )

    st.subheader("Udder sensor readings")
    st.caption("Enter readings in the same units and scale as the training dataset.")

    sensor_names = ["IUFL", "EUFL", "IUFR", "EUFR",
                    "IURL", "EURL", "IURR", "EURR"]
    sensors = {}
    cols = st.columns(4)
    for i, name in enumerate(sensor_names):
        with cols[i % 4]:
            sensors[name] = st.number_input(
                name, min_value=0.0, value=150.0, step=1.0
            )

    submitted = st.form_submit_button("Estimate mastitis risk")

if submitted:
    quadrant_totals = {
        q: sensors[inner] + sensors[outer]
        for q, (inner, outer) in QUADRANTS.items()
    }

    mean_quadrant = float(np.mean(list(quadrant_totals.values())))
    if mean_quadrant <= 0:
        st.error("Sensor readings must produce a positive mean quadrant reading.")
        st.stop()

    swelling_raw = float(np.mean(list(sensors.values())))
    asymmetry_raw = (
        max(quadrant_totals.values()) - min(quadrant_totals.values())
    ) / mean_quadrant

    # Use the scaler parameters fitted by data_prep.py on training rows.
    summary_path = ROOT / "data" / "processed" / "data_prep_summary.json"
    import json
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    scale = summary["scaler"]
    raw_cols = ["swelling_index_raw", "asymmetry_index_raw",
                "temperature", "months_pp"]
    raw_values = [swelling_raw, asymmetry_raw, temperature, months_pp]

    # Recover train-only min/max values saved by data_prep.py.
    data_min = scale["data_min"]
    data_max = scale["data_max"]
    if data_min is None or data_max is None:
        st.error(
            "Could not locate saved scaler parameters in data_prep_summary.json. "
            "Check the preprocessing output before using predictions."
        )
        st.stop()

    normalized = []
    for value, lo, hi in zip(raw_values, data_min, data_max):
        scaled = 0.0 if hi == lo else (value - lo) / (hi - lo)
        normalized.append(float(np.clip(scaled, 0.0, 1.0)))

    row = {
        "swelling_index": normalized[0],
        "asymmetry_index": normalized[1],
        "temperature_n": normalized[2],
        "months_pp_n": normalized[3],
        "breed_enc": int(breed == "Holstein"),
        "hardness": hardness,
        "pain": pain,
    }

    feature_names = EARLY_FEATURES if model_name == "Early-Warning" else FULL_FEATURES
    x = np.array([[row[name] for name in feature_names]], dtype=float)

    model = models[model_name]
    score = float(np.asarray(model.predict_score(x)).reshape(-1)[0])
    clipped_score = float(np.clip(score, 0.0, 1.0))
    tier = risk_tier(clipped_score)

    st.subheader("Prediction result")
    m1, m2 = st.columns(2)
    m1.metric("Model score (raw)", f"{score:.3f}")
    m2.metric("Risk category", str(tier))

    st.progress(clipped_score)
    st.write(f"Displayed score, clipped to 0–1: **{clipped_score:.1%}**")

    st.subheader("Feature values used")
    st.dataframe(pd.DataFrame([row])[feature_names], use_container_width=True)

    st.subheader("Fuzzy-rule explanation")
    try:
        explanation = explain_row(model, x[0], feature_names)
        st.json(explanation)
    except Exception as exc:
        st.warning(f"Explanation could not be generated: {exc}")

    st.warning(
        "Research prototype only. This score is not a veterinary diagnosis. "
        "Consult a qualified veterinarian for animal health decisions."
    )
