import numpy as np
import pandas as pd
import streamlit as st

from common import (badge, current_instance, data_badges, instance_tag, money, page_setup, plan_for, safe,
                    sidebar)

page_setup("Simulator & Replan")


def summarize_actions(plan) -> pd.Series:
    def fmt(g):
        parts = []
        for r in g.itertuples():
            s = f"{r.action} {r.quantity_kg:,.0f}"
            if isinstance(r.dest_id, str):
                s += f"→{r.dest_id}"
            if r.discount > 0:
                s += f" ({r.discount:.0%} off)"
            parts.append(s)
        return "; ".join(parts)
    return plan.rows.groupby("batch_id").apply(fmt, include_groups=False)


@st.cache_data(show_spinner="Simulating 40 realized worlds for both plans…")
def realized_compare(tag: str, _post, _stale, _new, _fifo, n: int = 40) -> pd.DataFrame:
    from freshroute.simulator import realize, simulate

    rows = []
    for s in range(5000, 5000 + n):
        w = realize(_post, s)
        for name, p in (("Replanned", _new), ("Stale plan (no replan)", _stale), ("FIFO baseline", _fifo)):
            m = simulate(_post, p, w)
            rows.append({"seed": s, "plan": name, "net_revenue": m["net_revenue"], "spoilage_pct": m["spoilage_pct"],
                         "fulfillment_pct": m["fulfillment_pct"]})
    return pd.DataFrame(rows)


def body():
    from freshroute.baseline import fifo
    from freshroute.checker import check_plan
    from freshroute.scenarios import apply_scenario, disruption_scenario

    ctl = sidebar()
    st.title("4 · Scenario simulator & live replan")
    inst, status = current_instance(ctl)
    st.markdown(data_badges(inst), unsafe_allow_html=True)
    risk, plan, viol = plan_for(inst, ctl["alpha"], instance_tag(ctl))

    st.markdown("Pick a disruption. FreshRoute re-runs the Monte Carlo and the optimizer and shows what changed and why.")
    lanes = sorted(inst.lanes)
    dests = sorted(inst.destinations)
    c = st.columns(5)
    with c[0]:
        delta = st.number_input("Heatwave +°C", 1.0, 15.0, 6.0, 0.5)
        if st.button("🔥 Heatwave", use_container_width=True):
            st.session_state.disruption = ("heatwave", {"delta_c": delta})
    with c[1]:
        lane_d = st.selectbox("Delayed lane", lanes, index=lanes.index("NSK-HYD") if "NSK-HYD" in lanes else 0)
        hrs = st.number_input("Delay hours", 1.0, 48.0, 8.0, 1.0)
        if st.button("🚚 Delayed truck", use_container_width=True):
            st.session_state.disruption = ("delayed_truck", {"lane": lane_d, "hours": hrs})
    with c[2]:
        lane_c = st.selectbox("Lane with fewer trucks", lanes, index=lanes.index("NSK-MUM") if "NSK-MUM" in lanes else 0)
        mult = st.slider("Capacity kept", 0.0, 1.0, 0.5, 0.1)
        if st.button("📉 Capacity drop", use_container_width=True):
            st.session_state.disruption = ("capacity_drop", {"lane": lane_c, "mult": mult})
    with c[3]:
        dest_s = st.selectbox("Market with demand spike", dests)
        sm = st.slider("Demand multiplier", 1.1, 2.5, 1.4, 0.1)
        if st.button("📈 Demand spike", use_container_width=True):
            st.session_state.disruption = ("demand_spike", {"dest": dest_s, "mult": sm})
    with c[4]:
        dest_o = st.selectbox("New order for", dests, index=dests.index("PUN") if "PUN" in dests else 0)
        coms = sorted({cc for (d, cc) in inst.demand if d == dest_o}) or sorted(inst.commodities)
        com_o = st.selectbox("Commodity", coms)
        kg_o = st.number_input("kg", 500.0, 20000.0, 3000.0, 500.0)
        if st.button("🧾 New order", use_container_width=True):
            st.session_state.disruption = ("new_order", {"dest": dest_o, "commodity": com_o, "kg": kg_o})
    if st.button("Reset"):
        st.session_state.pop("disruption", None)

    dis = st.session_state.get("disruption")
    if not dis:
        st.info("No disruption applied yet. Current plan: "
                f"{len(plan.shipments)} shipments, expected net {money(plan.objective)}.")
        return
    kind, kw = dis
    scen = disruption_scenario(inst, kind, seed=0, **kw)
    post = apply_scenario(inst, scen)
    st.markdown(badge(f"Applied: {scen.description}", "fallback"), unsafe_allow_html=True)
    tag = instance_tag(ctl) + "|" + scen.scenario_id + "|" + str(sorted(kw.items()))
    risk2, new, viol2 = plan_for(post, ctl["alpha"], tag)
    stale_viol = check_plan(post, plan, risk2)

    k = st.columns(4)
    k[0].metric("Expected net (replanned)", money(new.objective), money(new.objective - plan.objective) + " vs before")
    k[1].metric("Replanned violations", len(viol2))
    k[2].metric("Old plan violations under new conditions", len(stale_viol))
    k[3].metric("Re-solve time", f"{new.solve_time_s:.2f} s")
    if not stale_viol.empty:
        with st.expander("Why the old plan is no longer valid"):
            st.dataframe(stale_viol, hide_index=True, use_container_width=True)

    before, after = summarize_actions(plan), summarize_actions(new)
    diff = pd.DataFrame({"before": before, "after": after}).fillna("–")
    diff["changed"] = diff.before != diff.after
    st.subheader(f"What changed ({int(diff.changed.sum())} of {len(diff)} batches)")
    reasons = new.rows.groupby("batch_id").reason.apply(lambda r: " | ".join(r))
    diff["why (new plan)"] = reasons
    show_all = st.toggle("Show unchanged batches too", value=False)
    st.dataframe(diff if show_all else diff[diff.changed], use_container_width=True,
                 column_config={"why (new plan)": st.column_config.TextColumn(width="large")})

    st.subheader("Realized outcome on 40 sampled worlds of the disrupted scenario")
    comp = realized_compare(tag, post, plan, new, fifo(post))
    g = comp.groupby("plan")[["net_revenue", "spoilage_pct", "fulfillment_pct"]].agg(["mean", lambda x: np.percentile(x, 5),
                                                                                       lambda x: np.percentile(x, 95)])
    g.columns = [f"{a} {('mean' if b == 'mean' else ('p5' if b == '<lambda_0>' else 'p95'))}" for a, b in g.columns]
    st.dataframe(g.round(1), use_container_width=True)
    st.caption("Same 40 realized worlds (seeds 5000–5039) for every plan. The stale plan is the plan made before "
               "the disruption was known. The disruption suite on the Robustness page repeats this over 60 held-out seeds per type.")


safe(body)
