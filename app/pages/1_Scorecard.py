import pandas as pd
import plotly.express as px
import streamlit as st

from common import badge, load_records, load_scorecard, load_timing, money, page_setup, safe

page_setup("Scorecard")

COLORS = {"FreshRoute planner": "#1f7a5a", "Deterministic optimizer (ablation)": "#7a9cc6",
          "Baseline A: FIFO": "#c0603a", "Baseline B: greedy-by-price": "#d9a441"}


def body():
    st.title("1 · Scorecard: plan vs baseline")
    sc = load_scorecard()
    if not sc:
        st.info("Run `make reproduce` first; this page only shows numbers computed by that run.")
        return
    ss = sc["seed_sets"]
    st.markdown(
        badge(f"{sc['data_source']} DATA", "syn" if sc["data_source"] == "SYNTHETIC" else "org")
        + badge(f"HELD-OUT · {sc['heldout']['n_scenarios']} scenarios · seeds {ss['heldout'][0]}–{ss['heldout'][1]}", "ok")
        + badge(f"α = {sc['risk_alpha_tuned']} chosen on tuning seeds {ss['tuning'][0]}–{ss['tuning'][1]} only")
        + badge(f"cost model: {sc['cost_model_source']}"),
        unsafe_allow_html=True)
    if sc.get("quick_mode"):
        st.warning("This scorecard came from a --quick smoke run (small seed sets). Re-run `make reproduce` for headline numbers.")
    if sc["data_source"] == "SYNTHETIC":
        st.caption("SYNTHETIC DATA: results describe the model on a generated dataset, not real-world performance.")

    h = sc["heldout"]["summary"]
    tm = load_timing().get("heldout", {}).get("summary", {})
    rows = []
    for m, x in h.items():
        rows.append({
            "Method": m,
            "Spoilage % (mean)": x["spoilage_pct"]["mean"], "Spoilage % p5–p95": f"{x['spoilage_pct']['p5']:.1f}–{x['spoilage_pct']['p95']:.1f}",
            "Fulfilled % (mean)": x["fulfillment_pct"]["mean"], "Fulfilled % p5–p95": f"{x['fulfillment_pct']['p5']:.1f}–{x['fulfillment_pct']['p95']:.1f}",
            "Net revenue (mean)": money(x["net_revenue"]["mean"]), "Net revenue median": money(x["net_revenue"]["median"]),
            "Net p5–p95": f"{money(x['net_revenue']['p5'])} – {money(x['net_revenue']['p95'])}",
            "Diverted kg (mean)": round(x["diverted_kg"]["mean"]),
            "Min destination fill %": round(x["min_dest_fill_pct"]["mean"], 1),
            "Violations / plan": round(x["violations_mean"], 2),
            "Plans with 0 violations": f"{x['plans_with_zero_violations_pct']:.0f}%",
            "Solve time (mean s)": round(tm.get(m, {}).get("solve_time_s_mean", float("nan")), 3),
        })
    df = pd.DataFrame(rows)
    order = list(COLORS)
    df["o"] = df.Method.map({m: i for i, m in enumerate(order)})
    st.dataframe(df.sort_values("o").drop(columns="o").round(2), hide_index=True, use_container_width=True)

    st.subheader("Paired differences (same scenario, same realized world)")
    comp = pd.DataFrame(sc["heldout"]["paired_vs_baselines"])
    comp["95% CI"] = comp.apply(lambda r: f"[{r.ci95_low:,.2f}, {r.ci95_high:,.2f}]", axis=1)
    comp = comp[["baseline", "metric", "n", "mean_diff", "95% CI", "median_diff", "relative_change_pct", "win_rate_pct"]]
    comp.columns = ["Compared with", "Metric", "n", "Mean difference", "95% CI", "Median difference",
                    "Relative change %", "Planner better in % of scenarios"]
    st.dataframe(comp.round(2), hide_index=True, use_container_width=True)
    st.caption("Spoilage differences are percentage points (negative is better). Net revenue in INR per dispatch cycle. "
               "CI = mean ± 1.96 standard errors of the paired differences.")

    rec = load_records("heldout_records.csv")
    if rec is not None:
        st.subheader("Distribution across held-out scenarios")
        metric = st.selectbox("Metric", ["net_revenue", "spoilage_pct", "fulfillment_pct", "unmet_kg", "violations",
                                         "diverted_kg", "min_dest_fill_pct", "net_revenue_excl_carry"])
        fig = px.box(rec, x="method", y=metric, color="method", color_discrete_map=COLORS, points=False,
                     category_orders={"method": order})
        fig.update_layout(showlegend=False, height=380, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Unmet demand by destination (mean kg)")
        cols = [c for c in rec.columns if c.startswith("unmet_") and c != "unmet_kg"]
        um = rec.groupby("method")[cols].mean().rename(columns=lambda c: c.replace("unmet_", ""))
        st.dataframe(um.round(0), use_container_width=True)
        st.subheader("Constraint violations by type (mean per plan)")
        vcols = [c for c in rec.columns if c.startswith("violations_")]
        st.dataframe(rec.groupby("method")[vcols].mean().round(2), use_container_width=True)
        st.caption("Baselines are checked by the same independent checker with the same α. Their shelf-life "
                   "violations are shipments whose Monte Carlo P(sellable on arrival) is below 1−α; hold violations "
                   "are stock left at the hub that is likely to expire.")

    st.subheader("α tuning (tuning seeds only)")
    st.caption(sc.get("alpha_selection_rule", ""))
    st.dataframe(pd.DataFrame(sc["tuning_table"]).round(2), hide_index=True)


safe(body)
