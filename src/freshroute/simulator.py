"""One shared simulator for every method.

``realize(inst, seed)`` samples a single "realized" world: true remaining life per
batch, actual travel time and temperature per lane, realized demand and prices, hub
storage temperature. ``simulate(inst, plan, world)`` applies a plan to that world and
measures spoilage, fulfillment, allocation balance and net revenue using the cost
model in the instance. Planner and baselines are scored on exactly the same world.

Accounting rules (same for every method; see docs/ASSUMPTIONS.md):
  * handling loss fraction of every shipped kg is spoiled;
  * a shipment arriving with life < required life (by discount) is spoiled;
  * a shipment arriving after the delivery window is rejected and sold at salvage
    value (distress sale) unless the cost model says late = spoiled;
  * sellable stock is sold against realized demand, deepest discount first, then
    shortest life first; each kg sold at discount k uses 1/(1+elasticity*k) of base demand;
  * unsold stock at a destination is carried over if it still has >= 1 day more than
    the minimum sellable life, otherwise it is spoiled;
  * held stock is carried over if, after the hold, it still has enough life for the
    shortest lane plus the minimum sellable life, otherwise it is spoiled;
  * diverted stock earns salvage value;
  * net revenue = sales + salvage (divert + distress) + carry-over value - transport
    - handling - holding - spoilage cost (purchase value) - unmet-demand penalty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .economics import carry_value_per_kg, salvage_per_kg, transport_var_per_kg
from .schema import Instance
from .shelflife import consumed_days, rate_multiplier, required_life


@dataclass
class World:
    seed: int
    life: dict[str, float]
    travel: dict[str, float]
    temps: dict[str, np.ndarray]
    demand: dict[tuple[str, str], float]
    price: dict[tuple[str, str], float]
    store_temp: dict[str, float]


def realize(inst: Instance, seed: int) -> World:
    u = inst.settings.get("uncertainty", {})
    life = {}
    for j, b in enumerate(sorted(inst.batches, key=lambda x: x.batch_id)):
        rng = np.random.default_rng([seed, 101, j])
        s = inst.commodities[b.commodity].life_sigma * float(u.get("life_sigma_scale", 1.0))
        life[b.batch_id] = float(b.remaining_life_days * np.exp(rng.normal(-0.5 * s * s, s)))
    travel, temps = {}, {}
    for i, lid in enumerate(sorted(inst.lanes)):
        lane = inst.lanes[lid]
        rng = np.random.default_rng([seed, 102, i])
        s = float(u.get("travel_sigma", 0.12))
        t = lane.base_travel_hours * np.exp(rng.normal(-0.5 * s * s, s))
        if rng.random() < float(u.get("delay_prob", 0.1)):
            t += rng.exponential(float(u.get("delay_hours_mean", 5.0)))
        travel[lid] = float(t)
        fc = inst.temp_forecast[lid]
        sig = float(u.get("temp_sigma0", 1.0)) + float(u.get("temp_sigma_per_day", 1.5)) * np.arange(len(fc)) / 24.0
        temps[lid] = fc + rng.normal() * sig
    rng = np.random.default_rng([seed, 103])
    cv = float(u.get("demand_cv", 0.15))
    demand = {k: float(v * np.exp(rng.normal(-0.5 * cv * cv, cv))) for k, v in sorted(inst.demand.items())}
    rng = np.random.default_rng([seed, 104])
    pcv = float(u.get("price_cv", 0.08))
    price = {k: float(v * np.exp(rng.normal(-0.5 * pcv * pcv, pcv))) for k, v in sorted(inst.prices.items())}
    rng = np.random.default_rng([seed, 105])
    store = {h: float(hub.storage_temp_c + rng.normal(0, 1.5)) for h, hub in sorted(inst.hubs.items())}
    return World(seed, life, travel, temps, demand, price, store)


def _min_lane_need(inst: Instance, b) -> float:
    com = inst.commodities[b.commodity]
    handling = float(inst.settings.get("planning", {}).get("handling_hours", 3.0))
    needs = []
    for lid, lane in inst.lanes.items():
        if lane.origin == b.origin and lane.max_trucks > 0:
            temps = np.full(1, lane.reefer_temp_c) if lane.reefer else inst.temp_forecast[lid]
            needs.append(float(consumed_days(temps, lane.base_travel_hours + handling, com)[0]))
    return com.min_sellable_life_days + (min(needs) if needs else 0.0)


def simulate(inst: Instance, plan, world: World) -> dict:
    cm = inst.cost_model
    e = cm.discount.elasticity
    handling_h = float(inst.settings.get("planning", {}).get("handling_hours", 3.0))
    hold_days = float(inst.settings.get("planning", {}).get("hold_days", 1))
    rows = plan.rows
    batches = {b.batch_id: b for b in inst.batches}
    total_kg = sum(b.quantity_kg for b in inst.batches)

    spoiled = {"handling_loss": 0.0, "arrived_unsellable": 0.0, "unsold_at_destination": 0.0, "held_expired": 0.0}
    carried_kg = diverted_kg = distress_kg = shipped_kg = 0.0
    revenue = discount_given = salvage_rev = carry_val = transport = handling = holding = 0.0
    late_spoil = cm.late_delivery == "spoiled"
    spoil_by_com: dict[str, float] = {}

    def spoil(kind: str, commodity: str, kg: float) -> None:
        spoiled[kind] += kg
        spoil_by_com[commodity] = spoil_by_com.get(commodity, 0.0) + kg

    ship = rows[rows.action.isin(["SHIP", "DISCOUNT"])]
    sellable = []  # (dest, commodity, discount, arrival_life, kg, batch)
    for r in ship.itertuples():
        b = batches[r.batch_id]
        com = inst.commodities[b.commodity]
        lane = inst.lanes[r.lane_id]
        q = float(r.quantity_kg)
        shipped_kg += q
        t = world.travel[r.lane_id]
        if lane.reefer:
            cons = consumed_days(np.full(1, lane.reefer_temp_c), t, com)[0] + consumed_days(world.temps[r.lane_id], handling_h, com)[0]
        else:
            cons = consumed_days(world.temps[r.lane_id], t + handling_h, com)[0]
        A = world.life[r.batch_id] - cons
        loss = q * com.handling_loss_frac
        spoil("handling_loss", b.commodity, loss)
        good = q - loss
        if A < required_life(com, r.discount, cm.discount.sell_speedup):
            spoil("arrived_unsellable", b.commodity, good)
        elif t > lane.window_end_hours:
            if late_spoil:
                spoil("arrived_unsellable", b.commodity, good)
            else:
                distress_kg += good
                salvage_rev += good * salvage_per_kg(inst, b.commodity)
        else:
            sellable.append((r.dest_id, b.commodity, float(r.discount), A, good, r.batch_id))
        transport += q * transport_var_per_kg(inst, lane)
        handling += q * cm.handling_cost_per_kg
    for lid, g in ship.groupby("lane_id"):
        lane = inst.lanes[lid]
        transport += cm.transport.per_truck_fixed * math.ceil(g.quantity_kg.sum() / lane.truck_capacity_kg - 1e-9)

    demand_left = dict(world.demand)
    fulfilled = {k: 0.0 for k in world.demand}
    sold_kg = 0.0
    sellable.sort(key=lambda s: (s[0], s[1], -s[2], s[3]))
    for d, c, k, A, kg, bid in sellable:
        units_left = demand_left.get((d, c), 0.0)
        per_unit = 1.0 / (1.0 + e * k)
        sell = min(kg, units_left / per_unit) if per_unit > 0 else 0.0
        if sell > 0:
            demand_left[(d, c)] = units_left - sell * per_unit
            fulfilled[(d, c)] = fulfilled.get((d, c), 0.0) + sell * per_unit
            p = world.price.get((d, c), 0.0)
            revenue += sell * p * (1 - k)
            discount_given += sell * p * k
            sold_kg += sell
        rest = kg - sell
        if rest > 0:
            com = inst.commodities[c]
            if A - 1.0 >= com.min_sellable_life_days:
                carried_kg += rest
                carry_val += rest * carry_value_per_kg(inst, c)
            else:
                spoil("unsold_at_destination", c, rest)

    for r in rows[rows.action == "HOLD"].itertuples():
        b = batches[r.batch_id]
        com = inst.commodities[b.commodity]
        q = float(r.quantity_kg)
        holding += q * cm.holding_cost_per_kg_day * hold_days
        after = world.life[r.batch_id] - hold_days * float(rate_multiplier(world.store_temp[b.origin], com))
        if after >= _min_lane_need(inst, b):
            carried_kg += q
            carry_val += q * carry_value_per_kg(inst, b.commodity)
        else:
            spoil("held_expired", b.commodity, q)
    for r in rows[rows.action == "DIVERT"].itertuples():
        q = float(r.quantity_kg)
        diverted_kg += q
        salvage_rev += q * salvage_per_kg(inst, r.commodity)

    spoilage_cost = sum(kg * inst.commodities[c].purchase_price_per_kg for c, kg in spoil_by_com.items())
    total_demand = sum(world.demand.values())
    total_ful = sum(fulfilled.values())
    unmet = {k: world.demand[k] - fulfilled.get(k, 0.0) for k in world.demand}
    penalty = cm.unmet_demand_penalty_per_kg * sum(unmet.values())
    fill_by_dest = {}
    for d in inst.destinations:
        dem = sum(v for (dd, _), v in world.demand.items() if dd == d)
        if dem > 0:
            fill_by_dest[d] = sum(v for (dd, _), v in fulfilled.items() if dd == d) / dem
    fills = np.array(list(fill_by_dest.values())) if fill_by_dest else np.array([np.nan])
    spoiled_kg = sum(spoiled.values())
    net = revenue + salvage_rev + carry_val - transport - handling - holding - spoilage_cost - penalty
    return {
        "total_kg": total_kg, "shipped_kg": shipped_kg, "sold_kg": sold_kg, "spoiled_kg": spoiled_kg,
        "spoilage_pct": 100.0 * spoiled_kg / total_kg, "diverted_kg": diverted_kg,
        "spoiled_or_diverted_pct": 100.0 * (spoiled_kg + diverted_kg) / total_kg, "carried_kg": carried_kg,
        "distress_kg": distress_kg, **{f"spoiled_{k}_kg": v for k, v in spoiled.items()},
        "demand_kg": total_demand, "fulfilled_kg": total_ful,
        "fulfillment_pct": 100.0 * total_ful / total_demand if total_demand else float("nan"),
        "unmet_kg": total_demand - total_ful,
        "min_dest_fill_pct": 100.0 * float(np.nanmin(fills)), "dest_fill_std_pct": 100.0 * float(np.nanstd(fills)),
        "revenue": revenue, "discount_given": discount_given, "salvage_revenue": salvage_rev,
        "carry_value": carry_val, "transport_cost": transport, "handling_cost": handling,
        "holding_cost": holding, "spoilage_cost": spoilage_cost, "unmet_penalty": penalty,
        "net_revenue": net, "net_revenue_excl_carry": net - carry_val,
        "fill_by_dest": fill_by_dest, "unmet_by_dest": _by_dest(unmet), "spoiled_by_commodity": spoil_by_com,
    }


def _by_dest(m: dict) -> dict:
    out = {}
    for (d, _), v in m.items():
        out[d] = out.get(d, 0.0) + v
    return out
