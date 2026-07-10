"""
Fraud Loss Prevention Dashboard
Run with: streamlit run app.py
Requires scored_test_set.csv in the same folder (precomputed model scores).
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Loss Prevention Engine", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("scored_test_set.csv")

df = load_data()
total_fraud = int(df["y_true"].sum())
total_txns = len(df)

st.title("Credit Risk & Fraud Loss Prevention Engine")
st.caption(
    "Framed around a Credit One-style Risk / Fraud Strategy tradeoff: "
    "catching fraud at the lowest cost to good customers, not just maximizing detection."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    k = st.slider(
        "Review capacity (top-k flagged transactions)",
        min_value=10, max_value=3000, value=150, step=10
    )
with col2:
    cost_false_decline = st.number_input(
        "Assumed cost per false decline ($)", min_value=0, max_value=200, value=15
    )
with col3:
    cost_review = st.number_input(
        "Assumed manual review cost per flagged txn ($)", min_value=0, max_value=50, value=2
    )

topk = df.iloc[:k]
caught = int(topk["y_true"].sum())
fraud_dollars = float(topk.loc[topk["y_true"] == 1, "Amount"].sum())
false_positives = k - caught
precision = caught / k
recall = caught / total_fraud
net_savings = fraud_dollars - false_positives * cost_false_decline - k * cost_review

st.markdown("### Results at this review capacity")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Precision", f"{precision:.1%}")
m2.metric("Recall", f"{recall:.1%}")
m3.metric("Fraud $ caught", f"${fraud_dollars:,.0f}")
m4.metric("False positives", f"{false_positives:,}")
m5.metric("Net savings", f"${net_savings:,.0f}", delta=None)

st.markdown("---")
st.markdown("### Net savings across review-capacity budgets")

ks = list(range(20, 2500, 20))
rows = []
for kk in ks:
    tk = df.iloc[:kk]
    c = int(tk["y_true"].sum())
    d = float(tk.loc[tk["y_true"] == 1, "Amount"].sum())
    fp = kk - c
    ns = d - fp * cost_false_decline - kk * cost_review
    rows.append((kk, ns))
curve = pd.DataFrame(rows, columns=["k", "net_savings"])
best_row = curve.loc[curve["net_savings"].idxmax()]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(curve["k"], curve["net_savings"], color="#2c3e50")
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.axvline(k, color="#e74c3c", linestyle=":", linewidth=1.5, label=f"Current setting (k={k})")
ax.scatter([best_row["k"]], [best_row["net_savings"]], color="#27ae60", zorder=5,
           label=f"Optimal k={int(best_row['k'])} (${best_row['net_savings']:,.0f})")
ax.set_xlabel("Review budget (k)")
ax.set_ylabel("Net savings ($)")
ax.legend()
st.pyplot(fig)

st.info(
    f"At these cost assumptions, the optimal review capacity is **{int(best_row['k'])} transactions** "
    f"per {total_txns:,}-transaction window, projecting **${best_row['net_savings']:,.0f}** in net savings. "
    f"Beyond that point, additional review volume costs more in false-decline friction than it recovers in caught fraud."
)

st.caption(
    "Dataset: ULB/Worldline Credit Card Fraud Detection (test holdout, 30% of 284,807 transactions). "
    "Cost assumptions are illustrative — adjust the sliders to reflect your own business case."
)
