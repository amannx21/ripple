import streamlit as st

from common import badge, load_scorecard, money, page_setup, safe

page_setup("ROI")


def body():
    st.title("7 · ROI estimate")
    sc = load_scorecard()
    if not sc:
        st.info("Run `make reproduce` first: the ROI is derived only from computed held-out results.")
        return
    roi = sc["roi_estimate"]
    st.markdown(badge(f"{sc['data_source']} DATA", "syn" if sc["data_source"] == "SYNTHETIC" else "org") +
                badge(f"held-out n = {sc['heldout']['n_scenarios']}"), unsafe_allow_html=True)
    if sc["data_source"] == "SYNTHETIC":
        st.warning("Illustrative only: computed on SYNTHETIC data with placeholder cost and shelf-life parameters. "
                   "Not a forecast of real savings.")
    per_cycle = roi["planner_minus_fifo_net_inr_per_cycle_mean"]
    lo, hi = roi["ci95"]
    cycles = st.slider("Dispatch cycles per month", 4, 31, int(roi["dispatch_cycles_per_month"]))
    tonnes = st.number_input("Your stock per cycle (tonnes)", 5.0, 2000.0, float(roi["stock_per_cycle_tonnes"]), 5.0)
    scale = tonnes / roi["stock_per_cycle_tonnes"]
    c = st.columns(3)
    c[0].metric("Extra net revenue per cycle (mean)", money(per_cycle * scale))
    c[1].metric("Per month (mean)", money(per_cycle * scale * cycles))
    c[2].metric("Per month, 95% CI", f"{money(lo * scale * cycles)} – {money(hi * scale * cycles)}")
    h = sc["heldout"]["summary"]
    p, f = h["FreshRoute planner"], h["Baseline A: FIFO"]
    spoil_t = (f["spoiled_kg"]["mean"] - p["spoiled_kg"]["mean"]) / 1000 * scale * cycles
    st.metric("Produce saved from spoilage per month", f"{spoil_t:,.1f} t")
    st.markdown(f"""
**How this is computed**
- Paired difference in net revenue, FreshRoute minus FIFO, averaged over {sc['heldout']['n_scenarios']} held-out scenarios
  (same scenario and same realized world for both), with a 95% CI of the mean.
- Multiplied by dispatch cycles per month. Scaled linearly by stock volume relative to the
  {roi['stock_per_cycle_tonnes']} t of the evaluated dataset. Linear scaling is an assumption; capacity and demand would not scale exactly.
- Costs included: transport (per kg-km and per truck), handling, holding, spoilage at purchase value, salvage,
  carry-over value. See the Data & Assumptions page for every value.
- {roi['assumptions']}
""")


safe(body)
