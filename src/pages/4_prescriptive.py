"""Prescriptive view (UC4): HDU bed allocation for sepsis admissions.

Analysis and page by Qizheng (Kevin) Zhao. The analysis module is
`src/prescriptive/uc4_prescriptive.py`; tab 3 reads its saved outputs from
`results/uc4/` (run `python src/prescriptive/uc4_prescriptive.py` to refresh
them) and falls back to a short simulation if they are missing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prescriptive import uc4_prescriptive as uc4  # noqa: E402

BLUE, GREY, ORANGE = "#2a78d6", "#8a8986", "#eb6834"
st.title("Which sepsis patients should get the free HDU beds?")
st.caption("Intended user: the bed manager. Data: Sepsis-3 study cohort (Chicco & Jurman, 2020). "
           "Predicted risks are cross-fitted, so no patient is scored by a model trained on it. "
           "This is a teaching prototype, not a validated clinical tool.")


@st.cache_data(show_spinner="Fitting the risk model (5-fold cross-fitting)…")
def prepare():
    df = uc4.load_cohort(uc4.DEFAULT_DATA_DIR, "study")
    df["p"] = uc4.cross_fitted_risk(df)
    return df


df = prepare()
y, p = df["died"].to_numpy(), df["p"].to_numpy()
tab1, tab2, tab3 = st.tabs(["1 · Choose a risk threshold", "2 · Today's bed board", "3 · Compare policies"])

# ------------------------------------------------------------------ tab 1
with tab1:
    th = np.round(np.arange(0.05, 0.351, 0.01), 2)
    nb_m, nb_a = uc4.net_benefit(y, p, th)
    fig = go.Figure()
    fig.add_scatter(x=th, y=nb_m, name="Risk model", line=dict(color=BLUE, width=3))
    fig.add_scatter(x=th, y=np.clip(nb_a, -0.03, None), name="Monitor all", line=dict(color=ORANGE, dash="dash"))
    fig.add_scatter(x=th, y=0 * th, name="Monitor none", line=dict(color=GREY, dash="dot"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickformat=".0%",
                      xaxis_title="Risk threshold", yaxis_title="Net benefit", legend=dict(orientation="h", y=1.08))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    t = c2.slider("Threshold", 0.05, 0.30, 0.20, 0.01, format="%.2f")
    tab = uc4.threshold_table(y, p, [t]).iloc[0]
    c2.metric("Patients flagged per 1,000", f"{tab['Flagged per 1,000']}")
    c2.metric("Share of deaths among flagged", f"{tab['Deaths captured (%)']}%")
    c2.metric("False alarms per death captured", f"{tab['False alarms per death captured']}")
    c2.caption("A threshold of t means the hospital accepts (1 − t)/t false alarms for each correctly flagged death.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Set the state of the HDU this morning and draw today's sepsis admissions.")
    c = st.columns(5)
    n_today = c[0].number_input("Admissions today", 2, 20, 8)
    free_a = c[1].number_input("Free beds, bay A", 0, 3, 1)
    sex_a = c[2].selectbox("Bay A holds", ["empty", "men", "women"], index=1)
    free_b = c[3].number_input("Free beds, bay B", 0, 3, 1)
    sex_b = c[4].selectbox("Bay B holds", ["empty", "men", "women"], index=2)
    seed = st.number_input("Day (random seed)", 0, 9999, 3)
    to_sex = {"empty": None, "men": 0, "women": 1}
    ward = uc4.Ward()
    free = [int(free_a) if sex_a != "empty" else 3, int(free_b) if sex_b != "empty" else 3]
    sexes = [to_sex[sex_a], to_sex[sex_b]]
    idx = np.random.default_rng(int(seed)).choice(len(df), size=int(n_today), replace=False)
    cand = {k: df[c_].to_numpy()[idx] for k, c_ in (("p", "p"), ("sex", "sex"), ("age", "age"), ("died", "died"))}
    board = pd.DataFrame({"arrival": range(1, n_today + 1), "age": cand["age"],
                          "sex": np.where(cand["sex"] == 1, "F", "M"), "risk %": (100 * cand["p"]).round(1)})
    for pol in ("FCFS", "CP-SAT", "CP-SAT + equity"):
        chosen = dict(uc4.allocate(pol, cand, free, sexes, ward))
        board[pol] = [("bay " + "AB"[chosen[i]]) if i in chosen else "" for i in range(n_today)]
    st.dataframe(board.sort_values("risk %", ascending=False), hide_index=True, width="stretch")
    st.caption("CP-SAT: maximise Σ xᵢ·pᵢ subject to free beds per bay and single-sex bays. The equity version also "
               "keeps at least 15% of HDU beds over the period for patients under 65 (shown here for one day).")

# ------------------------------------------------------------------ tab 3
with tab3:
    res = uc4.DEFAULT_OUT_DIR / "policy_summary.csv"
    if res.exists():
        ps = pd.read_csv(res)
        st.caption("From `uc4_prescriptive.py`: 200 simulated 90-day periods, 6 HDU beds, 8 admissions per day.")
    else:
        with st.spinner("Running a short simulation (20 periods)…"):
            ps = uc4.paired_summary(uc4.simulate(df, uc4.POLICIES, n_runs=20), rrr=0.15)
    rrr = st.slider("Assumed relative risk reduction of HDU care", 0.05, 0.30, 0.15, 0.05)
    show = ps[["policy", "deaths_in_HDU_per_1000", "vs_FCFS", "under65_share_%", "occupancy_%"]].copy()
    show["deaths averted per 1,000 (at chosen RRR)"] = (rrr * show["deaths_in_HDU_per_1000"]).round(2)
    show.columns = ["Policy", "Deaths in HDU per 1,000", "Difference vs FCFS", "Under 65 in HDU (%)",
                    "Occupancy (%)", "Deaths averted per 1,000"]
    st.dataframe(show.round(1), hide_index=True, width="stretch")
    st.info("RRR is an assumption: the data contain no treatment information. It rescales every policy by the "
            "same factor, so the ranking of the policies does not depend on it.")
