import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (badge, current_instance, instance_tag, load_records, load_scorecard, page_setup, plan_for, safe,
                    sidebar)

page_setup("Robustness & Sensitivity")
COLORS = {"FreshRoute planner": "#1f7a5a", "Baseline A: FIFO": "#c0603a", "Stale plan (no replan)": "#8c8c8c",
          "Deterministic optimizer (ablation)": "#7a9cc6", "Baseline B: greedy-by-price": "#d9a441"}


def body():
    ctl = sidebar()
    st.title("5 · Robustness & sensitivity")
    sc = load_scorecard()
    if sc:
        st.markdown(badge(f"{sc['data_source']} DATA", "syn" if sc["data_source"] == "SYNTHETIC" else "org"),
                    unsafe_allow_html=True)

    st.subheader("Monte Carlo view of one decision")
    inst, _ = current_instance(ctl)
    risk, plan, _ = plan_for(inst, ctl["alpha"], instance_tag(ctl))
    pairs = sorted(risk.samples)
    opts = [f"{b} → {d}" for b, d in pairs]
    default = 0
    if not plan.shipments.empty:
        s0 = plan.shipments.iloc[0]
        key = f"{s0.batch_id} → {s0.dest_id}"
        default = opts.index(key) if key in opts else 0
    sel = st.selectbox("Batch → destination", opts, index=default)
    b, d = pairs[opts.index(sel)]
    A = risk.samples[(b, d)]
    com = inst.commodities[inst.batch(b).commodity]
    fig = px.histogram(x=A, nbins=60, labels={"x": "remaining life on arrival (days)"})
    fig.add_vline(x=com.min_sellable_life_days, line_color="#8a1f17", annotation_text="min sellable")
    fig.update_layout(height=300, yaxis_title=f"samples (of {len(A)})")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"P(arrival life < {com.min_sellable_life_days} d) = **{np.mean(A < com.min_sellable_life_days):.1%}** · "
                f"5th–95th percentile {np.percentile(A, 5):.2f}–{np.percentile(A, 95):.2f} d. "
                "Sources of spread: lognormal shelf-life estimate error, travel-time noise and delays, temperature forecast error "
                "growing with horizon.")

    sens = load_records("sensitivity_records.csv")
    if sens is None:
        st.info("Run `make reproduce` to compute the sensitivity and disruption suites.")
        return
    st.subheader("Sensitivity: the world gets noisier than the planner assumes")
    st.caption("Seeds 3000–3059 (reporting only). The planner keeps its default assumptions; only the realized world's "
               "noise level changes. This is a model-misspecification stress test.")
    factor = st.selectbox("Factor", list(dict.fromkeys(sens.factor)))
    s = sens[sens.factor == factor].groupby(["level", "method"])[["net_revenue", "spoilage_pct", "fulfillment_pct"]].mean().reset_index()
    c = st.columns(3)
    for col, (m, t) in zip(c, [("net_revenue", "net revenue (INR)"), ("spoilage_pct", "spoilage %"),
                               ("fulfillment_pct", "fulfilled %")]):
        f = px.line(s, x="level", y=m, color="method", markers=True, color_discrete_map=COLORS)
        f.update_layout(height=280, yaxis_title=t, showlegend=(m == "net_revenue"), legend=dict(orientation="h", y=-0.3))
        col.plotly_chart(f, use_container_width=True)

    dis = load_records("disruption_records.csv")
    if dis is not None:
        st.subheader("Disruption suite (60 held-out seeds per disruption, disruption known at planning time)")
        g = dis.groupby(["disruption", "method"])[["net_revenue", "spoilage_pct", "fulfillment_pct", "violations"]].mean().reset_index()
        f = px.bar(g, x="disruption", y="net_revenue", color="method", barmode="group", color_discrete_map=COLORS)
        f.update_layout(height=330, yaxis_title="mean net revenue (INR)")
        st.plotly_chart(f, use_container_width=True)
        st.dataframe(g.round(2), hide_index=True, use_container_width=True)
        st.caption("Stale plan = the optimizer's plan made before the disruption, executed anyway. Its 'violations' "
                   "column counts constraints it breaks under the new conditions.")

    held = load_records("heldout_records.csv")
    if held is not None:
        st.subheader("Robustness across 500 held-out scenarios")
        p = held.pivot_table(index="seed", columns="method", values="net_revenue")
        if "FreshRoute planner" in p and "Baseline A: FIFO" in p:
            d = p["FreshRoute planner"] - p["Baseline A: FIFO"]
            f = px.histogram(d, nbins=50, labels={"value": "planner − FIFO net revenue per scenario (INR)"})
            f.add_vline(x=0, line_color="#333")
            f.update_layout(height=280, showlegend=False)
            st.plotly_chart(f, use_container_width=True)
            st.markdown(f"Planner better in **{100 * (d > 0).mean():.1f}%** of held-out scenarios; worst case "
                        f"{d.min():,.0f} INR, median {d.median():,.0f} INR.")
        if "disruptions" in held.columns:
            hd = held.assign(disrupted=held.disruptions.fillna("").ne("")).groupby(["disrupted", "method"])[["net_revenue", "spoilage_pct"]].mean()
            st.dataframe(hd.round(2), use_container_width=True)
            st.caption("~30% of held-out scenarios include a random disruption (heatwave, delay, capacity drop, demand spike, new order).")
    if sc:
        st.subheader("α tuning curve (tuning seeds 0–99 only)")
        t = pd.DataFrame(sc["tuning_table"])
        f = go.Figure([go.Scatter(x=t.alpha, y=t.net_revenue, mode="lines+markers", name="net revenue", yaxis="y1",
                                  line=dict(color="#1f7a5a")),
                       go.Scatter(x=t.alpha, y=t.spoilage_pct, mode="lines+markers", name="spoilage %", yaxis="y2",
                                  line=dict(color="#c0603a"))])
        f.update_layout(height=300, xaxis_title="α", yaxis=dict(title="mean net revenue"),
                        yaxis2=dict(title="spoilage %", overlaying="y", side="right"))
        f.add_vline(x=sc["risk_alpha_tuned"], line_dash="dot")
        st.plotly_chart(f, use_container_width=True)
        st.caption(sc.get("alpha_selection_rule", ""))


safe(body)
