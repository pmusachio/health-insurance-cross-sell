"""Interactive propensity-ranking dashboard.

Scores a single customer's likelihood to buy vehicle insurance and shows where they
fall in the contact priority list, plus a campaign view (cumulative gains and lift
at a chosen contact capacity) computed on the versioned sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="Cross-Sell Propensity Ranking", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(config.SAMPLE_PATH) if config.SAMPLE_PATH.exists() else pd.DataFrame()


@st.cache_data
def sample_scores() -> np.ndarray:
    df = load_sample()
    if df.empty:
        return np.array([])
    return load_predictor().score(df)


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.yaxis.label.set_color(D["foreground"])
    ax.title.set_color(D["foreground"])
    ax.grid(True, color=D["current_line"], linestyle="--", alpha=0.4)


def gains_chart(scores: np.ndarray, y: np.ndarray, capacity_pct: int):
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]
    cum = np.cumsum(y_sorted) / y_sorted.sum()
    pct = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=D["background"])
    ax.plot(pct * 100, cum * 100, color=D["green"], linewidth=2, label="Model")
    ax.plot([0, 100], [0, 100], color=D["comment"], linestyle="--", linewidth=1.5, label="Random")
    ax.axvline(capacity_pct, color=D["pink"], linestyle=":", linewidth=1.5)
    ax.set_xlabel("Customers contacted (%)")
    ax.set_ylabel("Interested buyers captured (%)")
    ax.legend(facecolor=D["current_line"], edgecolor=D["comment"], labelcolor=D["foreground"], fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    st.title("Health Insurance Cross-Sell — Propensity Ranking")
    st.markdown(
        "Scores how likely a customer is to buy vehicle insurance, so a limited sales team "
        "contacts the most promising customers first."
    )

    with st.sidebar:
        st.header("Customer")
        age = st.slider("Age", 18, 90, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        previously_insured = st.selectbox("Previously insured", [0, 1],
                                          format_func=lambda v: "Yes" if v else "No")
        vehicle_age = st.selectbox("Vehicle age", list(config.VEHICLE_AGE_ORDER), index=1)
        vehicle_damage = st.selectbox("Past vehicle damage", ["Yes", "No"])
        annual_premium = st.number_input("Annual premium", 2000.0, 100000.0, 30000.0, 500.0)
        vintage = st.slider("Vintage (days as customer)", 10, 300, 150)
        driving_license = st.selectbox("Driving license", [1, 0],
                                       format_func=lambda v: "Yes" if v else "No")
        region_code = st.number_input("Region code", 0.0, 52.0, 28.0, 1.0)
        channel = st.number_input("Policy sales channel", 1.0, 163.0, 26.0, 1.0)
        run = st.button("Score customer", type="primary")

    record = {
        "Age": age, "Gender": gender, "Driving_License": driving_license,
        "Region_Code": region_code, "Previously_Insured": previously_insured,
        "Vehicle_Age": vehicle_age, "Vehicle_Damage": vehicle_damage,
        "Annual_Premium": annual_premium, "Policy_Sales_Channel": channel, "Vintage": vintage,
    }

    if run:
        score = predictor.score_one(record)
        ref = sample_scores()
        pct = predictor.rank_percentile(score, ref)
        st.subheader("Customer score")
        c = st.columns(3)
        c[0].metric("Propensity to buy", f"{score*100:.1f}%")
        c[1].metric("Contact priority (top)", f"{100 - pct:.0f}%")
        c[2].metric("Base rate", f"{predictor.base_rate*100:.1f}%")
        verdict = ("high priority" if pct >= 80 else "medium priority" if pct >= 50 else "low priority")
        st.markdown(
            f"This customer ranks above {pct:.0f}% of the reference base, so they are a "
            f"**{verdict}** contact. A score of {score*100:.1f}% versus a {predictor.base_rate*100:.1f}% "
            f"base rate means they are {score/predictor.base_rate:.1f}x more likely to convert than average."
        )

        st.subheader("Most influential features (model-wide)")
        imp = pd.DataFrame(predictor.top_features(6)).rename(
            columns={"feature": "Feature", "importance": "Permutation importance (ROC AUC drop)"})
        st.dataframe(imp, hide_index=True, width="stretch")

    df = load_sample()
    if not df.empty and config.TARGET in df.columns:
        st.subheader("Campaign view (reference sample)")
        capacity = st.slider("Contact capacity (% of base)", 5, 100, config.DEFAULT_CAPACITY_PCT, 5)
        scores = sample_scores()
        y = df[config.TARGET].to_numpy()
        order = np.argsort(scores)[::-1]
        k = int(len(y) * capacity / 100)
        captured = y[order][:k].sum() / y.sum() if y.sum() else 0
        lift = (y[order][:k].mean() / y.mean()) if (k and y.mean()) else 0
        left, right = st.columns([2, 1])
        with left:
            st.pyplot(gains_chart(scores, y, capacity))
        with right:
            st.metric("Buyers captured", f"{captured*100:.0f}%")
            st.metric("Lift vs random", f"{lift:.2f}x")
            st.caption(f"Contacting the top {capacity}% reaches {captured*100:.0f}% of buyers.")


if __name__ == "__main__":
    main()
