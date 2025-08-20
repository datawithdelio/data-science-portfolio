# app/streamlit_app.py
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path("models/laptop_price_pipeline.joblib")

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("💻 Laptop Price Predictor")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()
pipe = bundle["pipeline"]
feature_names = bundle["feature_names"]

st.success(f"✅ Model loaded with {len(feature_names)} features")
st.caption(f"Features: {feature_names}")

# Separate numeric vs binary
NUMERIC_FEATS = ["RAM size", "drive memory size (GB)"]
BINARY_FEATS = [f for f in feature_names if f not in NUMERIC_FEATS]

# ---------------- Upload ----------------
st.header("📂 Upload CSV or JSON")
uploaded = st.file_uploader("Upload file", type=["csv", "json"])

def load_any(f):
    if f.name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_json(f, orient="columns")

if uploaded:
    df_up = load_any(uploaded)
    st.subheader("Preview")
    st.dataframe(df_up.head())

    # align to training features
    X_new = df_up.reindex(columns=feature_names, fill_value=0).copy()
    for c in NUMERIC_FEATS:
        if c in X_new:
            X_new[c] = pd.to_numeric(X_new[c], errors="coerce").fillna(0)

    preds = pipe.predict(X_new)
    out = df_up.copy()
    out["Predicted Price"] = preds

    st.subheader("Predictions")
    st.dataframe(out.head(20))
    st.success(f"✅ Mean predicted price: ${preds.mean():,.0f}")

# ---------------- Manual Input ----------------
st.header("✍️ Manual Input")

manual = {}
# Numeric fields
col1, col2 = st.columns(2)
manual["RAM size"] = col1.number_input("RAM size (GB)", min_value=0.0, value=16.0, step=1.0)
manual["drive memory size (GB)"] = col2.number_input("Drive size (GB)", min_value=0.0, value=512.0, step=32.0)

# Binary features as checkboxes (0 or 1)
cols = st.columns(3)
for i, feat in enumerate(BINARY_FEATS):
    manual[feat] = 1 if cols[i % 3].checkbox(feat, value=False) else 0

if st.button("Predict manually"):
    row = pd.DataFrame([manual], columns=feature_names)
    pred = float(pipe.predict(row)[0])
    st.success(f"💰 Estimated Price: **${pred:,.0f}**")

# ---------------- Download Template ----------------
st.header("📥 Download a CSV template")
if st.button("Generate template"):
    template = pd.DataFrame([{c: (0 if c in BINARY_FEATS else 0.0) for c in feature_names}])
    st.download_button(
        label="Download template.csv",
        data=template.to_csv(index=False),
        file_name="template.csv",
        mime="text/csv",
    )