# app/app.py — Airbnb Market Gap • Dublin (tailored + robust)
# Uses your columns exactly; cleans list-like strings and guards sliders.

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Airbnb Market Gap • Dublin", layout="wide")
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "merged_airbnb.csv"

# ---------- helpers ----------
def clean_listlike_cell(x: str):
    """
    Cells like ',Ballsbridge,...' or ',Entire home/apt' appear in your data.
    - Split on comma
    - Drop empty tokens
    - Return the FIRST non-empty trimmed token
    """
    if pd.isna(x):
        return np.nan
    s = str(x)
    # if it looks like a list, split; otherwise just strip
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]  # remove blanks
    return parts[0] if parts else np.nan

@st.cache_data
def load_data(path: Path):
    if not path.exists():
        st.error(f"Missing data file: {path}")
        st.stop()
    df = pd.read_csv(path)

    # ---- Parse dates you actually have ----
    for c in [
        "ts_contact_at","ts_reply_at","ts_accepted_at","ts_booking_at",
        "ds_checkin_x","ds_checkout_x","ds_checkout_y"
    ]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # ---- Map to app's fields (your exact names) ----
    # Main date for filtering/plots
    df.rename(columns={"ds_checkin_x": "date_main"}, inplace=True)

    # Demand (searches)
    if "n_searches" in df.columns:
        df["count"] = pd.to_numeric(df["n_searches"], errors="coerce").fillna(0)
    else:
        df["count"] = 1  # fallback

    # Guests
    if "n_guests" in df.columns:
        df["n_guests"] = pd.to_numeric(df["n_guests"], errors="coerce")

    # Price for filter/KPI (use filter_price_max)
    if "filter_price_max" in df.columns:
        df["classification_max_price"] = pd.to_numeric(df["filter_price_max"], errors="coerce")

    # Country / Neighborhood / Room type — clean leading commas
    if "origin_country" in df.columns:
        df["origin_country"] = df["origin_country"].astype(str).str.strip()
    if "filter_neighborhoods" in df.columns:
        df["filter_neighborhoods"] = df["filter_neighborhoods"].apply(clean_listlike_cell)
    if "filter_room_types" in df.columns:
        df["filter_room_types"] = df["filter_room_types"].apply(clean_listlike_cell)

    # Acceptance: non-null ts_accepted_at => accepted
    df["accepted_flag"] = df["ts_accepted_at"].notna().astype(int) if "ts_accepted_at" in df.columns else 0
    df["acceptance_rate"] = df["accepted_flag"].astype(float)  # row-level; KPI uses mean

    # Lead time (days) if we have contact + checkin
    if {"ts_contact_at","date_main"}.issubset(df.columns):
        df["lead_time_days"] = (df["date_main"] - df["ts_contact_at"]).dt.days

    # Simple gap score
    rng = df["count"].max() - df["count"].min()
    cnt_norm = (df["count"] - df["count"].min()) / (rng if rng != 0 else 1)
    df["gap_score"] = (1 - df["acceptance_rate"].clip(0,1)) * 0.7 + cnt_norm * 0.3

    return df

df = load_data(DATA_PATH)

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")

# Reset button
if st.sidebar.button("Reset filters"):
    st.experimental_rerun()

# Date range
if "date_main" in df.columns:
    min_d, max_d = pd.to_datetime(df["date_main"].min()), pd.to_datetime(df["date_main"].max())
    d1, d2 = st.sidebar.date_input("Date range", value=(min_d, max_d))
else:
    d1 = d2 = None

# Only show multiselects if data exists
def ms(label, col):
    if col in df.columns and df[col].notna().any():
        opts = sorted(df[col].dropna().unique().tolist())
        return st.sidebar.multiselect(label, opts, default=[])
    return []

country_sel = ms("Origin country", "origin_country")
neigh_sel   = ms("Neighborhood",   "filter_neighborhoods")
room_sel    = ms("Room type",      "filter_room_types")

# Price control — robust bounds
price_max = None
if "classification_max_price" in df.columns and df["classification_max_price"].notna().any():
    # Use sane quantiles; cap slider at p99 (or 2000 if p99 is tiny) to avoid insane max values
    p95 = float(np.nanpercentile(df["classification_max_price"], 95))
    p99 = float(np.nanpercentile(df["classification_max_price"], 99))
    default_max = int(p95) if np.isfinite(p95) and p95 > 0 else 1000
    slider_cap = int(max(200, min(p99, 10000)))  # keep it reasonable
    price_max = st.sidebar.number_input("Max price (€)", min_value=0, value=min(default_max, slider_cap), max_value=slider_cap)

# Guests slider
guests = None
if "n_guests" in df.columns and df["n_guests"].notna().any():
    gmax = int(max(2, np.nanmax(df["n_guests"])))
    guests = st.sidebar.slider("Guests (≤)", 1, gmax, min(2, gmax))

# ---------------- Apply filters ----------------
q = df.copy()
if d1 and d2 and "date_main" in q.columns:
    q = q[(q["date_main"] >= pd.to_datetime(d1)) & (q["date_main"] <= pd.to_datetime(d2))]
if country_sel:
    q = q[q["origin_country"].isin(country_sel)]
if neigh_sel:
    q = q[q["filter_neighborhoods"].isin(neigh_sel)]
if room_sel:
    q = q[q["filter_room_types"].isin(room_sel)]
if price_max is not None and "classification_max_price" in q.columns:
    q = q[q["classification_max_price"] <= price_max]
if guests is not None and "n_guests" in q.columns:
    q = q[q["n_guests"] <= guests]

# ---------------- Header ----------------
st.title("Airbnb Market Gap — Dublin")
st.caption("Filters use your exact columns. If nothing shows up, click ‘Reset filters’ or widen them.")

# If no rows, show a helpful warning and stop rendering
if q.empty:
    st.warning("No rows match the current filters. Try clearing selections or increasing Max price / Guests.")
    st.stop()

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
acc = q["acceptance_rate"].mean() if "acceptance_rate" in q.columns else np.nan
c1.metric("Acceptance Rate", f"{acc*100:0.1f}%" if pd.notna(acc) else "—")

searches = int(q["count"].sum()) if "count" in q.columns else len(q)
c2.metric("Searches", f"{searches:,}")

lead = q["lead_time_days"].mean() if "lead_time_days" in q.columns else np.nan
c3.metric("Avg Lead Time", f"{lead:0.1f} days" if pd.notna(lead) else "—")

premium_share = np.nan
if "classification_max_price" in q.columns:
    premium_share = 100 * (q["classification_max_price"] >= 600).mean()
c4.metric("Premium (≥€600) Share", f"{premium_share:0.1f}%" if pd.notna(premium_share) else "—")

st.divider()

# ---------------- Charts ----------------
# Demand over time
if {"date_main","count"}.issubset(q.columns) and q["count"].sum() > 0:
    st.subheader("Demand over time")
    c = alt.Chart(q.dropna(subset=["date_main", "count"])).mark_line().encode(
        x=alt.X("date_main:T", title="Check-in date"),
        y=alt.Y("count:Q", title="Searches")
    ).properties(height=280)
    st.altair_chart(c, use_container_width=True)

# Acceptance by country
if {"origin_country","acceptance_rate"}.issubset(q.columns) and q["origin_country"].notna().any():
    st.subheader("Acceptance rate by country")
    g = q.groupby("origin_country", dropna=True)["acceptance_rate"].mean().reset_index()
    chart2 = alt.Chart(g).mark_bar().encode(
        x=alt.X("acceptance_rate:Q", axis=alt.Axis(format="%"), title="Acceptance rate"),
        y=alt.Y("origin_country:N", sort="-x", title=None)
    ).properties(height=360)
    st.altair_chart(chart2, use_container_width=True)

# Neighborhood table (uses accepted_flag from ts_accepted_at)
if {"filter_neighborhoods","count"}.issubset(q.columns):
    st.subheader("Neighborhood demand vs acceptance (Top 25 low-acceptance)")
    g2 = q.groupby("filter_neighborhoods").agg(
        searches=("count","sum"),
        accepted=("accepted_flag","sum")
    ).reset_index()
    g2["acceptance_rate"] = np.where(g2["searches"] > 0, g2["accepted"] / g2["searches"], np.nan)
    if "classification_max_price" in q.columns:
        p75 = q.groupby("filter_neighborhoods")["classification_max_price"].quantile(0.75)
        g2 = g2.merge(p75.rename("price_p75"), on="filter_neighborhoods", how="left")
    g2 = g2.sort_values(["acceptance_rate","searches"], ascending=[True, False], na_position="last")
    st.dataframe(g2.head(25), use_container_width=True)

# ---------------- Download ----------------
st.download_button(
    "Download filtered data (CSV)",
    q.to_csv(index=False).encode("utf-8"),
    "airbnb_filtered.csv",
)
