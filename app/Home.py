import streamlit as st

from common import base_instance, badge, data_badges, load_scorecard, money, page_setup, safe

page_setup("Home")


def body():
    inst = base_instance()
    st.title("FreshRoute Live")
    st.markdown("**Heat-aware, uncertainty-aware dispatch planning for perishable produce** · project *ripple*")
    st.markdown(data_badges(inst), unsafe_allow_html=True)
    st.markdown(
        "Remaining shelf life is not a fixed number. It is a probability distribution that shrinks faster in the heat. "
        "FreshRoute samples that distribution along every route (Monte Carlo), computes the **Spoilage-at-Risk (SaR)** "
        "of each batch-destination pair, and solves a risk-limited optimization that decides for every batch: "
        "**Ship · Discount · Hold · Divert**. Every recommendation comes with a plain-language reason and a counterfactual."
    )
    sc = load_scorecard()
    if sc:
        h = sc["heldout"]["summary"]
        p, f = h["FreshRoute planner"], h["Baseline A: FIFO"]
        st.subheader(f"Held-out results · {sc['heldout']['n_scenarios']} scenarios · seeds "
                     f"{sc['seed_sets']['heldout'][0]}–{sc['seed_sets']['heldout'][1]}")
        st.markdown(badge(f"{sc['data_source']} DATA", "syn" if sc["data_source"] == "SYNTHETIC" else "org") +
                    badge(f"α = {sc['risk_alpha_tuned']} (tuned on seeds {sc['seed_sets']['tuning'][0]}–{sc['seed_sets']['tuning'][1]})"),
                    unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("Spoilage (mean % of stock)", f"{p['spoilage_pct']['mean']:.1f}%",
                    f"{p['spoilage_pct']['mean'] - f['spoilage_pct']['mean']:+.1f} pts vs FIFO", delta_color="inverse")
        c[1].metric("Demand fulfilled (mean)", f"{p['fulfillment_pct']['mean']:.1f}%",
                    f"{p['fulfillment_pct']['mean'] - f['fulfillment_pct']['mean']:+.1f} pts vs FIFO")
        c[2].metric("Net revenue per cycle (mean)", money(p["net_revenue"]["mean"]),
                    f"{money(p['net_revenue']['mean'] - f['net_revenue']['mean'])} vs FIFO")
        st.caption("All numbers are read from results/scorecard.json, written by `make reproduce`. "
                   + ("They come from a SYNTHETIC dataset and are not real-world performance." if sc["data_source"] == "SYNTHETIC" else ""))
    else:
        st.info("No results yet. Run `make reproduce` (or `python -m freshroute.reproduce`) to compute the scorecard.")
    st.markdown("""
**Pages**
1. **Scorecard**: planner vs baselines on held-out scenarios
2. **Plan**: dispatch queue, SaR gauges, constraint check
3. **Live Map & Weather**: real routes, along-route temperature, source badges
4. **Simulator & Replan**: heatwave, delays, capacity, demand, new orders
5. **Robustness & Sensitivity**: Monte Carlo spread and stress tests
6. **Explain**: decision cards with counterfactuals
7. **ROI**: rupees per month from computed results
8. **Data & Assumptions**: schema, cost model, every assumption and limitation
""")


safe(body)
