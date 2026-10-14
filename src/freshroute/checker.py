"""Independent constraint checker.

Validates any plan (optimizer or baseline) using only the instance, the plan rows
and a Monte Carlo risk table. It does not reuse any optimizer internals, so a
"0 violations" badge means the plan really satisfies:

  stock          quantities per batch sum to the batch quantity, none negative
  lane           a shipment uses a lane that exists from the batch's origin to its destination
  trucks         kg on a lane <= trucks used x truck capacity, trucks used <= trucks available
  receiving      kg into a destination <= its receiving capacity
  demand         sellable kg (in base-demand units) <= forecast demand per destination/commodity
  discount       discount is one of the allowed levels
  window         P(arrival after the delivery window) <= alpha
  shelf_life     P(arrival life >= required sellable life and on time) >= 1 - alpha
  hold           held stock has P(still sellable next cycle) >= 1 - alpha
"""
from __future__ import annotations

import math

import pandas as pd

from .montecarlo import RiskTable
from .schema import Instance

TOL_KG = 1.0


def check_plan(inst: Instance, plan, risk: RiskTable, alpha: float | None = None) -> pd.DataFrame:
    alpha = plan.alpha if alpha is None else alpha
    rows = plan.rows
    v = []

    def add(kind, batch=None, where=None, detail=""):
        v.append({"constraint": kind, "batch_id": batch, "where": where, "detail": detail})

    batches = {b.batch_id: b for b in inst.batches}
    for bid, b in batches.items():
        q = rows.loc[rows.batch_id == bid, "quantity_kg"]
        if (q < -1e-9).any():
            add("stock", bid, None, "negative quantity")
        if abs(q.sum() - b.quantity_kg) > TOL_KG:
            add("stock", bid, None, f"allocated {q.sum():,.1f} kg of {b.quantity_kg:,.1f} kg")
    unknown = set(rows.batch_id) - set(batches)
    for bid in unknown:
        add("stock", bid, None, "plan references an unknown batch")

    ship = rows[rows.action.isin(["SHIP", "DISCOUNT"])].copy()
    levels = inst.cost_model.discount.levels
    for r in ship.itertuples():
        b = batches.get(r.batch_id)
        lane = inst.lanes.get(r.lane_id)
        if b is None:
            continue
        if lane is None or lane.origin != b.origin or lane.dest_id != r.dest_id:
            add("lane", r.batch_id, r.dest_id, f"no lane {b.origin}->{r.dest_id} as '{r.lane_id}'")
        if not any(abs(r.discount - k) < 1e-9 for k in levels):
            add("discount", r.batch_id, r.dest_id, f"discount {r.discount} not in {levels}")
        if r.action == "SHIP" and r.discount > 0 or r.action == "DISCOUNT" and r.discount <= 0:
            add("discount", r.batch_id, r.dest_id, "action label does not match discount")

    for lid, g in ship.groupby("lane_id"):
        lane = inst.lanes.get(lid)
        if lane is None:
            continue
        kg = g.quantity_kg.sum()
        needed = math.ceil(max(kg - TOL_KG, 0) / lane.truck_capacity_kg)
        used = plan.trucks.get(lid, needed)
        if used > lane.max_trucks:
            add("trucks", None, lid, f"{used} trucks used, {lane.max_trucks} available")
        if kg > used * lane.truck_capacity_kg + TOL_KG:
            add("trucks", None, lid, f"{kg:,.0f} kg on {used} trucks of {lane.truck_capacity_kg:,.0f} kg")

    for d, g in ship.groupby("dest_id"):
        cap = inst.destinations[d].receiving_capacity_kg
        if g.quantity_kg.sum() > cap + TOL_KG:
            add("receiving", None, d, f"{g.quantity_kg.sum():,.0f} kg > capacity {cap:,.0f} kg")

    e = inst.cost_model.discount.elasticity
    if not ship.empty:
        ship["units"] = [r.quantity_kg * (1 - inst.commodities[r.commodity].handling_loss_frac) / (1 + e * r.discount)
                         for r in ship.itertuples()]
        for (d, c), g in ship.groupby(["dest_id", "commodity"]):
            D = inst.demand_of(d, c)
            if g.units.sum() > D + TOL_KG:
                add("demand", None, f"{d}/{c}", f"{g.units.sum():,.0f} kg-equivalent > demand {D:,.0f} kg")

    pr = risk.pairs.set_index(["batch_id", "dest_id", "discount"])
    for r in ship.itertuples():
        key = (r.batch_id, r.dest_id, float(r.discount))
        if key not in pr.index:
            add("shelf_life", r.batch_id, r.dest_id, "no risk estimate for this option")
            continue
        o = pr.loc[key]
        if o.p_late_any > alpha + 1e-9:
            add("window", r.batch_id, r.dest_id, f"P(late) {o.p_late_any:.0%} > {alpha:.0%}")
        if o.p_ok < 1 - alpha - 1e-9:
            add("shelf_life", r.batch_id, r.dest_id, f"P(sellable on arrival) {o.p_ok:.0%} < {1 - alpha:.0%}")

    hold = risk.hold.set_index("batch_id")
    for r in rows[rows.action == "HOLD"].itertuples():
        pv = float(hold.loc[r.batch_id, "p_viable_after_hold"])
        if pv < 1 - alpha - 1e-9:
            add("hold", r.batch_id, None, f"P(still sellable next cycle) {pv:.0%} < {1 - alpha:.0%}")
    return pd.DataFrame(v, columns=["constraint", "batch_id", "where", "detail"])
