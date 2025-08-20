# app.py — polished UI, Predict button (no live updates), clear explanations
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------- Config ----------
DATA_PATH = Path("data/Data.csv")
MODEL_PATH = Path("models/best_pipeline.pkl")

# Hide identifiers from the form (the model may still expect them if trained with them)
HIDE_IN_FORM = {"phone_number"}

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
.main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {
  padding: 1.2rem 1.6rem; border-radius: 14px;
  background: linear-gradient(135deg, #151a28 0%, #0f1220 100%);
  border: 1px solid #252a3a; box-shadow: 0 6px 18px rgba(0,0,0,.25);
}
.hero h1 {margin: 0 0 .2rem 0; font-weight: 700;}
.hero p {opacity: .7; margin: .2rem 0 0 0}
.card { padding: 1.0rem 1.2rem; border-radius: 14px; background: #0e1117; border: 1px solid #222739; }
.metric { font-size: 40px; font-weight: 800; line-height: 1; margin: .1rem 0; }
.subtle {opacity: .65}
.stProgress > div > div > div { background-image: linear-gradient(90deg,#5b8cff,#8d7dff) !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider("Decision threshold (churn if prob ≥ threshold)", 0.1, 0.9, 0.50, 0.05)
st.sidebar.caption("Click **Predict** to update the results.")

# ---------- Helpers ----------
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_sample():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        for t in ["churn","exited","target","label","y"]:
            if t in df.columns:
                return df.drop(columns=[t]), t
    return pd.DataFrame(), None

def get_feature_groups(pre):
    num_cols, cat_cols = [], []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    return num_cols, cat_cols

def expected_cols(pre):
    return get_feature_groups(pre)

def fill_missing_expected(row: pd.DataFrame, pre, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Ensure all training columns exist at inference."""
    num_all, cat_all = expected_cols(pre)

    for col in num_all:
        if col not in row.columns:
            val = float(X_sample[col].median()) if not X_sample.empty and col in X_sample.columns else 0.0
            row[col] = val

    for col in cat_all:
        if col not in row.columns:
            if not X_sample.empty and col in X_sample.columns and not X_sample[col].dropna().empty:
                val = str(X_sample[col].dropna().mode().iloc[0])
            else:
                val = ""
            row[col] = val

    order = [*(num_all + cat_all)]
    cols = [c for c in order if c in row.columns] + [c for c in row.columns if c not in order]
    return row[cols]

def predict_proba(pipe, row: pd.DataFrame) -> float:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return float(pipe.predict_proba(row)[0, 1])
    if hasattr(clf, "decision_function"):
        z = float(clf.decision_function(pipe.named_steps["pre"].transform(row))[0])
        return 1.0 / (1.0 + np.exp(-z))
    return float(pipe.predict(row)[0])

# ---------- Load artifacts ----------
pipe = load_pipeline()
X_sample, target_col = load_sample()
pre = pipe.named_steps["pre"]
clf = pipe.named_steps["clf"]
num_cols_all, cat_cols_all = get_feature_groups(pre)

# Form lists (hide identifiers)
num_cols = [c for c in num_cols_all if c not in HIDE_IN_FORM]
cat_cols = [c for c in cat_cols_all if c not in HIDE_IN_FORM]

# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <h1>Customer Churn Predictor</h1>
  <p>Enter customer attributes or upload a CSV to score at scale. Click <b>Predict</b> to update results.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ---------- Inputs ----------
st.subheader("Input")
left, right = st.columns(2)

with left:
    st.markdown("##### Numeric features")
    nvals = {}
    for col in num_cols:
        if not X_sample.empty and col in X_sample.columns:
            s = X_sample[col].dropna()
            mn, mx = float(s.quantile(0.01)), float(s.quantile(0.99))
            val = float(s.median())
        else:
            mn, mx, val = 0.0, 99999.0, 0.0
        nvals[col] = st.number_input(col, value=val, min_value=mn, max_value=mx, key=f"num_{col}")

with right:
    st.markdown("##### Categorical features")
    cvals = {}
    for col in cat_cols:
        if not X_sample.empty and col in X_sample.columns:
            choices = sorted(map(str, X_sample[col].dropna().unique().tolist()))
            cvals[col] = st.selectbox(col, choices, index=0 if choices else None, key=f"cat_{col}")
        else:
            cvals[col] = st.text_input(col, "", key=f"cat_{col}")

# ---------- Predict button ----------
predict_clicked = st.button("🔮 Predict", use_container_width=True)
if not predict_clicked:
    st.stop()

# ---------- Build row & predict ----------
row = pd.DataFrame([{**nvals, **cvals}])
row = fill_missing_expected(row, pre, X_sample)

prob = predict_proba(pipe, row)
pred = int(prob >= threshold)

c1, c2, c3 = st.columns([1.1, 2, 2])
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Prediction")
    st.markdown(f'<div class="metric">{ "Churn" if pred==1 else "No Churn" }</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">label</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Probability of churn")
    st.markdown(f'<div class="metric">{prob:.3f}</div>', unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, prob)))
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Decision threshold")
    st.markdown(f'<div class="metric">{threshold:.2f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Adjust in the sidebar</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------- Explainability (active one-hot labels) ----------
st.subheader("Why this prediction?")
explained = False
try:
    import shap
    X_row = pre.transform(row)
    names = pre.get_feature_names_out()

    # Active OHE features for this row
    active_ohe = {f"{col}_{str(row[col].iloc[0])}" for col in cat_cols_all if col in row.columns}

    if hasattr(clf, "get_booster"):  # tree-based (XGBoost, etc.)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_row)
        vals = sv[0] if isinstance(sv, list) else sv
        contrib = vals[0]

        renamed, keep_mask = [], []
        for n in names:
            parts = n.split("_", 1)
            if len(parts) == 2 and parts[0] in cat_cols_all:
                keep = n in active_ohe
                keep_mask.append(keep)
                renamed.append(f"{parts[0]}=={parts[1]}" if keep else None)
            else:
                keep_mask.append(True)
                renamed.append(n)

        top = (
            pd.DataFrame({"feature": renamed, "impact": np.abs(contrib)[keep_mask]})
            .dropna(subset=["feature"])
            .sort_values("impact", ascending=False)
            .head(10)
            .set_index("feature")
        )
        st.bar_chart(top)
        explained = True
except Exception:
    pass

if not explained:
    # Fallback: global importances; still show only the active OHE category
    names = pre.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        renamed = []
        keep_vals = []
        for n in names:
            parts = n.split("_", 1)
            if len(parts) == 2 and parts[0] in cat_cols_all:
                active = f"{parts[0]}_{str(row[parts[0]].iloc[0])}"
                if n == active:
                    renamed.append(f"{parts[0]}=={parts[1]}")
                    keep_vals.append(True)
                else:
                    renamed.append(None)
                    keep_vals.append(False)
            else:
                renamed.append(n)
                keep_vals.append(True)
        imps = pd.Series(np.array(getattr(clf, "feature_importances_")), index=names)
        out = imps[keep_vals]
        out.index = [r for r, k in zip(renamed, keep_vals) if k]
        st.bar_chart(out.sort_values(ascending=False).head(10).rename("importance"))
    else:
        st.caption("Install `shap` for per-prediction explanations: `pip install shap`")

st.divider()

# ---------- Batch predictions ----------
st.subheader("Batch predictions")
st.caption("Upload a CSV with the same feature columns used in training.")
csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
if csv:
    df_in = pd.read_csv(csv)
    df_in.columns = [c.strip().lower().replace(" ", "_") for c in df_in.columns]
    df_in = fill_missing_expected(df_in, pre, X_sample)
    probs = (pipe.predict_proba(df_in)[:, 1] if hasattr(clf, "predict_proba") else pipe.predict(df_in).astype(float))
    preds = (probs >= threshold).astype(int)
    out = df_in.copy()
    out["churn_prob"] = probs
    out["churn_pred"] = preds
    st.dataframe(out.head(25))
    st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", use_container_width=True)
