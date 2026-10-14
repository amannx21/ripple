import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from common import (badge, current_instance, data_badges, instance_tag, page_setup, plan_for, safe, sidebar,
                    source_kind)

page_setup("Live Map & Weather")


@st.cache_data(show_spinner="Scoring departure hours…")
def departure_scan(tag: str, _inst, _plan, hours: tuple) -> pd.DataFrame:
    """Shipment-weighted SaR of the current plan's shipments if trucks left at another hour (same routes)."""
    from freshroute.montecarlo import compute_risk
    from freshroute.temperature import build_forecasts

    ship = _plan.shipments
    out = []
    for h in hours:
        inst = copy.copy(_inst)
        inst.settings = copy.deepcopy(_inst.settings)
        inst.settings["planning"]["departure_hour"] = h
        if _inst.temp_source.startswith("SYNTHETIC") or _inst.temp_source.startswith("FALLBACK"):
            inst.temp_forecast = build_forecasts(inst, h)
        else:  # live forecast: shift the along-route series by the hour difference (same day)
            shift = (h - _inst.settings["planning"]["departure_hour"]) % 24
            inst.temp_forecast = {k: np.roll(v, -shift) for k, v in _inst.temp_forecast.items()}
        r = compute_risk(inst, n=500).pairs.set_index(["batch_id", "dest_id", "discount"])
        sar = [r.loc[(s.batch_id, s.dest_id, s.discount), "p_bad"] for s in ship.itertuples()]
        w = ship.quantity_kg.to_numpy()
        out.append({"departure_hour": h, "weighted_p_not_sellable": float(np.dot(sar, w) / w.sum()) if w.sum() else 0.0})
    return pd.DataFrame(out)


def body():
    ctl = sidebar()
    st.title("3 · Live map & weather")
    inst, status = current_instance(ctl)
    st.markdown(data_badges(inst), unsafe_allow_html=True)
    if status is None:
        st.info("Offline mode: lanes use the dataset's distances and the synthetic diurnal temperature model. "
                "Switch the sidebar to **Live geography** for OSRM routes and Open-Meteo forecasts "
                "(cached, rate-limited, with automatic fallback).")
    else:
        st.markdown(badge(f"weather: {status['weather']}", source_kind(status["weather"])) +
                    badge("routing: OSRM demo server (no live traffic)", "neutral") +
                    badge("prices: REPLAY (not live)", "neutral"), unsafe_allow_html=True)
        for n in status["notes"]:
            st.caption(n)

    risk, plan, viol = plan_for(inst, ctl["alpha"], instance_tag(ctl))
    paths = status["paths"] if status else {lid: [[inst.hubs[l.origin].lat, inst.hubs[l.origin].lon],
                                                   [inst.destinations[l.dest_id].lat, inst.destinations[l.dest_id].lon]]
                                             for lid, l in inst.lanes.items()}
    kg_lane = plan.shipments.groupby("lane_id").quantity_kg.sum()
    route_rows = []
    for lid, path in paths.items():
        lane = inst.lanes[lid]
        tmax = float(np.max(inst.temp_forecast[lid][: int(np.ceil(lane.base_travel_hours)) + 1]))
        heat = min(1.0, max(0.0, (tmax - 20) / 20))
        route_rows.append({"lane": lid, "path": [[p[1], p[0]] for p in path], "kg": float(kg_lane.get(lid, 0)),
                           "color": [int(60 + 190 * heat), int(150 - 100 * heat), 60, 220 if kg_lane.get(lid, 0) else 90],
                           "width": 3 + 6 * float(kg_lane.get(lid, 0)) / max(1.0, float(kg_lane.max() if len(kg_lane) else 1)),
                           "tmax": round(tmax, 1)})
    nodes = [{"name": h.name, "lon": h.lon, "lat": h.lat, "kind": "hub", "color": [31, 122, 90]} for h in inst.hubs.values()]
    nodes += [{"name": d.name, "lon": d.lon, "lat": d.lat, "kind": "market", "color": [192, 96, 58]} for d in inst.destinations.values()]
    layers = [
        pdk.Layer("PathLayer", data=route_rows, get_path="path", get_color="color", get_width="width",
                  width_units="pixels", pickable=True),
        pdk.Layer("ScatterplotLayer", data=nodes, get_position="[lon, lat]", get_fill_color="color",
                  get_radius=18000, pickable=True),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=17.0, longitude=76.5, zoom=4.6),
                             tooltip={"text": "{lane}{name}\n{kg} kg planned · max {tmax} C on trip"}),
                    use_container_width=True)
    st.caption("Line colour = hottest forecast hour on the trip (green cool → red hot). Line width = kg planned on the lane.")

    tbl = []
    for lid, lane in inst.lanes.items():
        row = {"lane": lid, "distance km": round(lane.distance_km, 1), "travel h": round(lane.base_travel_hours, 1),
               "reefer": lane.reefer, "kg planned": float(kg_lane.get(lid, 0))}
        if status:
            s = status["lanes"][lid]
            row["route source"] = s["route_source"]
            row["temperature source"] = s.get("temp_source", "")
        else:
            row["route source"] = "dataset"
            row["temperature source"] = inst.temp_source
        tbl.append(row)
    st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

    st.subheader("Temperature along the route")
    lid = st.selectbox("Lane", sorted(inst.lanes))
    lane = inst.lanes[lid]
    H = int(np.ceil(lane.base_travel_hours)) + 6
    t = inst.temp_forecast[lid][:H]
    u = inst.settings["uncertainty"]
    sig = u["temp_sigma0"] + u["temp_sigma_per_day"] * np.arange(H) / 24
    hrs = np.arange(H)
    fig = go.Figure([
        go.Scatter(x=np.r_[hrs, hrs[::-1]], y=np.r_[t + 1.645 * sig, (t - 1.645 * sig)[::-1]], fill="toself",
                   fillcolor="rgba(192,96,58,0.15)", line=dict(width=0), name="90% forecast band"),
        go.Scatter(x=hrs, y=t, line=dict(color="#c0603a"), name="ambient forecast"),
    ])
    if lane.reefer:
        fig.add_hline(y=lane.reefer_temp_c, line_dash="dot", annotation_text=f"reefer {lane.reefer_temp_c} C")
    fig.add_vline(x=lane.base_travel_hours, line_dash="dash", annotation_text="expected arrival")
    fig.update_layout(height=320, xaxis_title=f"hours after departure ({ctl['departure_hour']}:00)",
                      yaxis_title="°C")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simulated in-transit temperature sensor")
    st.markdown(badge("SIMULATED SENSOR (not a real device)", "fallback"), unsafe_allow_html=True)
    ship_l = plan.shipments[plan.shipments.lane_id == lid]
    if ship_l.empty:
        st.caption("No shipment planned on this lane.")
    else:
        from freshroute.shelflife import consumed_days

        bsel = st.selectbox("Truck carrying batch", sorted(ship_l.batch_id.unique()))
        b = inst.batch(bsel)
        com = inst.commodities[b.commodity]
        seed = st.number_input("Sensor seed", 0, 9999, 7)
        rng = np.random.default_rng(int(seed))
        n = int(np.ceil(lane.base_travel_hours * 4)) + 1  # 15-minute readings
        tt = np.arange(n) / 4.0
        amb = np.interp(tt, np.arange(len(inst.temp_forecast[lid])), inst.temp_forecast[lid])
        drift = np.cumsum(rng.normal(0, 0.25, n))
        reading = (np.full(n, lane.reefer_temp_c) if lane.reefer else amb) + drift + rng.normal(0, 0.3, n)
        hourly = reading[::4]
        life = [b.remaining_life_days - consumed_days(hourly, h, com)[0] for h in tt]
        fig_s = go.Figure([go.Scatter(x=tt, y=reading, name="sensor °C", line=dict(color="#c0603a")),
                           go.Scatter(x=tt, y=life, name="remaining life (d)", yaxis="y2", line=dict(color="#1f7a5a"))])
        fig_s.add_hline(y=com.min_sellable_life_days, line_dash="dot", line_color="#1f7a5a", yref="y2",
                        annotation_text="min sellable")
        fig_s.update_layout(height=300, xaxis_title="hours in transit", yaxis_title="°C",
                            yaxis2=dict(title="days", overlaying="y", side="right"))
        st.plotly_chart(fig_s, use_container_width=True)
        hot = float(np.mean(reading > (com.ref_temp_c + 10)))
        msg = (f"Arrival life on this simulated trip: **{life[-1]:.2f} d** (min sellable {com.min_sellable_life_days} d). "
               f"{hot:.0%} of readings above {com.ref_temp_c + 10:.0f} °C.")
        if life[-1] < com.min_sellable_life_days:
            st.error("ALERT: this truck is now projected to arrive unsellable. Re-route to a nearer market or "
                     "prepare a markdown/divert on arrival.  \n" + msg)
        else:
            st.success(msg)

    st.subheader("Cooler-hours departure window")
    if plan.shipments.empty:
        st.info("No shipments in the current plan.")
    else:
        scan = departure_scan(instance_tag(ctl) + f"|{ctl['alpha']}", inst, plan, tuple(range(0, 24, 2)))
        best = scan.sort_values("weighted_p_not_sellable").iloc[0]
        fig2 = go.Figure(go.Bar(x=scan.departure_hour, y=100 * scan.weighted_p_not_sellable, marker_color="#7a9cc6"))
        fig2.update_layout(height=260, xaxis_title="departure hour", yaxis_title="kg-weighted P(not sellable) %")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"Leaving at **{int(best.departure_hour):02d}:00** gives the lowest kg-weighted spoilage risk "
                    f"for today's planned shipments ({100 * best.weighted_p_not_sellable:.1f}%). "
                    "Delivery windows are measured from departure, so check market opening times before changing.")

    st.subheader("Price replay and 7-day forecast baseline")
    from freshroute.live.prices import forecast_next, load_replay

    hist, label = load_replay()
    st.markdown(badge(label, "neutral"), unsafe_allow_html=True)
    com = st.selectbox("Commodity", sorted(hist.commodity.unique()))
    h = hist[hist.commodity == com].groupby("date").price_per_kg.mean().iloc[-120:]
    fc = forecast_next(hist, com)
    fig3 = go.Figure([go.Scatter(x=h.index, y=h.values, name="replayed price", line=dict(color="#333")),
                      go.Scatter(x=fc.date, y=fc.forecast, name="seasonal-naive forecast", line=dict(color="#1f7a5a")),
                      go.Scatter(x=np.r_[fc.date, fc.date[::-1]], y=np.r_[fc.hi, fc.lo[::-1]], fill="toself",
                                 fillcolor="rgba(31,122,90,0.15)", line=dict(width=0), name="80% band")])
    fig3.update_layout(height=300, yaxis_title="INR/kg")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Forecast = same day last year × (recent 14-day level / last year's level). A transparent baseline, "
               "not a trained model. Prices are not scraped and not live.")


safe(body)
