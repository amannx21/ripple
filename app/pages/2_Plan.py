import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (badge, current_instance, data_badges, instance_tag, money, page_setup, plan_for, safe,
                    sidebar)

page_setup("Plan")


def gauge(value: float, title: str, alpha: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * value, number={"suffix": "%", "font": {"size": 22}},
        title={"text": title, "font": {"size": 12}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, 100 * alpha], "color": "#cdeedd"},
                         {"range": [100 * alpha, 50], "color": "#fde7c8"},
                         {"range": [50, 100], "color": "#f6c7c2"}],
               "threshold": {"line": {"color": "#8a1f17", "width": 3}, "value": 100 * alpha}}))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=40, b=0))
    return fig


def body():
    ctl = sidebar()
    st.title("2 · Dispatch plan")
    inst, status = current_instance(ctl)
    st.markdown(data_badges(inst), unsafe_allow_html=True)
    risk, plan, viol = plan_for(inst, ctl["alpha"], instance_tag(ctl))
    if plan.status != "Optimal":
        st.error(f"Solver status: {plan.status}. No plan shown.")
        return
    v_badge = badge("0 violations ✓ (independent checker)", "ok") if viol.empty else badge(f"{len(viol)} violations", "bad")
    st.markdown(v_badge + badge(f"α = {plan.alpha:.2f}") + badge(f"solved in {plan.solve_time_s:.2f} s (HiGHS MILP)")
                + badge(f"{risk.n_samples} Monte Carlo samples per pair"), unsafe_allow_html=True)
    if not viol.empty:
        st.dataframe(viol, hide_index=True)

    rows = plan.rows
    kg = rows.groupby("action").quantity_kg.sum()
    total = sum(b.quantity_kg for b in inst.batches)
    c = st.columns(5)
    for i, a in enumerate(["SHIP", "DISCOUNT", "HOLD", "DIVERT"]):
        c[i].metric(a.title(), f"{kg.get(a, 0) / 1000:,.1f} t", f"{100 * kg.get(a, 0) / total:.0f}% of stock",
                    delta_color="off")
    c[4].metric("Expected net revenue", money(plan.objective))

    st.subheader("Dispatch queue (priority: least remaining life per hour of transit leaves first)")
    show = rows[["priority_rank", "batch_id", "commodity", "origin", "action", "dest_id", "discount", "quantity_kg",
                 "sar", "p_ok", "exp_net_inr", "reason"]].copy()
    show.columns = ["Rank", "Batch", "Commodity", "Hub", "Action", "Destination", "Discount", "kg", "SaR",
                    "P(ok)", "Expected net ₹", "Reason"]
    st.dataframe(show, hide_index=True, use_container_width=True, height=420,
                 column_config={"Discount": st.column_config.NumberColumn(format="%.0f%%"),
                                "SaR": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "P(ok)": st.column_config.NumberColumn(format="%.2f"),
                                "Reason": st.column_config.TextColumn(width="large")})
    st.download_button("Download dispatch sheet (CSV)", rows.to_csv(index=False).encode(), "dispatch_sheet.csv",
                       "text/csv")
    st.caption("SaR = P(arrival life < minimum sellable life) for the undiscounted option on that route. P(ok) = P(arrives "
               "with enough life for its discount level AND inside the delivery window). Hold rows show P(still sellable "
               "next cycle).")

    st.subheader("Spoilage-at-Risk gauges (per batch, best feasible destination)")
    st.caption(f"Green zone ≤ α. A batch with no destination in the green zone is held or diverted.")
    p0 = risk.pairs[risk.pairs.discount == 0]
    best = p0.sort_values("sar").groupby("batch_id").head(1).sort_values("sar", ascending=False)
    cols = st.columns(5)
    for i, r in enumerate(best.itertuples()):
        act = ", ".join(sorted(set(rows[rows.batch_id == r.batch_id].action)))
        cols[i % 5].plotly_chart(gauge(r.sar, f"{r.batch_id} {r.commodity} → {r.dest_id}<br>{act}", plan.alpha),
                                 use_container_width=True, key=f"g{r.batch_id}")

    st.subheader("Demand allocation and unmet demand (forecast)")
    e = inst.cost_model.discount.elasticity
    ship = plan.shipments.copy()
    ship["units"] = [q * (1 - inst.commodities[c].handling_loss_frac) / (1 + e * k)
                     for q, c, k in zip(ship.quantity_kg, ship.commodity, ship.discount)]
    alloc = ship.groupby(["dest_id", "commodity"]).units.sum()
    dem = pd.Series(inst.demand)
    dem.index.names = ["dest_id", "commodity"]
    tbl = pd.DataFrame({"demand_kg": dem, "allocated_kg_equiv": alloc}).fillna(0)
    tbl["unmet_kg"] = (tbl.demand_kg - tbl.allocated_kg_equiv).clip(lower=0)
    tbl["fill_%"] = 100 * tbl.allocated_kg_equiv / tbl.demand_kg.where(tbl.demand_kg > 0)
    st.dataframe(tbl.round(1), use_container_width=True)
    by_dest = tbl.groupby(level=0)[["demand_kg", "allocated_kg_equiv", "unmet_kg"]].sum()
    fig = go.Figure([go.Bar(name="allocated", x=by_dest.index, y=by_dest.allocated_kg_equiv, marker_color="#1f7a5a"),
                     go.Bar(name="unmet", x=by_dest.index, y=by_dest.unmet_kg, marker_color="#c0603a")])
    fig.update_layout(barmode="stack", height=300, yaxis_title="kg (base-demand equivalent)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trucks per lane")
    tr = pd.DataFrame([{"lane": l, "trucks used": plan.trucks.get(l, 0), "available": lane.max_trucks,
                        "kg loaded": float(plan.shipments[plan.shipments.lane_id == l].quantity_kg.sum()),
                        "capacity kg": lane.max_trucks * lane.truck_capacity_kg} for l, lane in inst.lanes.items()])
    st.dataframe(tr, hide_index=True)


safe(body)
