"""Scenario generation and application.

A scenario perturbs the *forecast-side* instance the planner sees (demand, prices,
temperatures, capacities, known delays, inventory). The *realized* outcome is
sampled separately by the simulator from the same seed, so planner and baselines
always face identical scenarios.
"""
from __future__ import annotations

import copy

import numpy as np

from .schema import Batch, DemandRow, Instance, Lane, Scenario
from .temperature import build_forecasts

DISRUPTIONS = ["heatwave", "delayed_truck", "capacity_drop", "demand_spike", "new_order"]


def _key_mult(m: dict[str, float], d: str, c: str) -> float:
    return m.get(f"{d}|{c}", 1.0) * m.get(f"{d}|*", 1.0) * m.get(f"*|{c}", 1.0) * m.get("*", 1.0)


def apply_scenario(base: Instance, scen: Scenario) -> Instance:
    inst = copy.deepcopy(base)
    inst.scenario = scen
    lanes = {}
    for lid, lane in inst.lanes.items():
        mult = scen.capacity_multiplier.get(lid, 1.0) * scen.capacity_multiplier.get("*", 1.0)
        upd = {"max_trucks": int(np.floor(lane.max_trucks * mult + 1e-9))}
        delay = scen.lane_delay_hours.get(lid, 0.0) + scen.lane_delay_hours.get("*", 0.0)
        if delay:
            upd["base_travel_hours"] = lane.base_travel_hours + delay
        lanes[lid] = lane.model_copy(update=upd)
    inst.lanes = lanes
    inst.demand = {(d, c): v * _key_mult(scen.demand_multiplier, d, c) for (d, c), v in inst.demand.items()}
    for order in scen.new_orders:
        inst.demand[(order.dest_id, order.commodity)] = inst.demand.get((order.dest_id, order.commodity), 0.0) + order.demand_kg
        if (order.dest_id, order.commodity) not in inst.prices:
            inst.prices[(order.dest_id, order.commodity)] = _fallback_price(inst, order.commodity)
    inst.prices = {(d, c): v * _key_mult(scen.price_multiplier, d, c) for (d, c), v in inst.prices.items()}
    batches = []
    for b in inst.batches:
        lm = scen.batch_life_multiplier.get(b.batch_id, 1.0) * scen.batch_life_multiplier.get("*", 1.0)
        qm = scen.batch_qty_multiplier.get(b.batch_id, 1.0) * scen.batch_qty_multiplier.get("*", 1.0)
        batches.append(b.model_copy(update={
            "remaining_life_days": round(b.remaining_life_days * lm, 3),
            "quantity_kg": max(100.0, round(b.quantity_kg * qm, -1)),
        }))
    inst.batches = batches
    # add the scenario's temperature offsets on top of whatever forecast the base carries
    # (synthetic diurnal model offline, Open-Meteo along-route series in live mode)
    if not base.temp_forecast:
        base_fc = build_forecasts(base)
    else:
        base_fc = base.temp_forecast
    inst.temp_forecast = {lid: np.asarray(base_fc[lid]) + scen.temp_offset_c + scen.lane_temp_offset_c.get(lid, 0.0)
                          for lid in inst.lanes}
    return inst


def _fallback_price(inst: Instance, commodity: str) -> float:
    vals = [v for (d, c), v in inst.prices.items() if c == commodity]
    return float(np.mean(vals)) if vals else 0.0


def disruption_scenario(base: Instance, kind: str, seed: int = 0, **kw) -> Scenario:
    """Named disruption presets used by the simulator buttons and the disruption suite."""
    rng = np.random.default_rng(seed)
    lanes = sorted(base.lanes)
    dests = sorted(base.destinations)
    if kind == "heatwave":
        return Scenario(scenario_id=f"heatwave-{seed}", seed=seed, temp_offset_c=kw.get("delta_c", 6.0),
                        description=f"Heatwave: +{kw.get('delta_c', 6.0)} C on every lane", disruptions=["heatwave"])
    if kind == "delayed_truck":
        lane = kw.get("lane") or lanes[int(rng.integers(len(lanes)))]
        hrs = kw.get("hours", 8.0)
        return Scenario(scenario_id=f"delay-{lane}-{seed}", seed=seed, lane_delay_hours={lane: hrs},
                        description=f"Known delay of {hrs} h on lane {lane}", disruptions=["delayed_truck"])
    if kind == "capacity_drop":
        lane = kw.get("lane") or lanes[int(rng.integers(len(lanes)))]
        return Scenario(scenario_id=f"capdrop-{lane}-{seed}", seed=seed, capacity_multiplier={lane: kw.get("mult", 0.5)},
                        description=f"Truck capacity on {lane} cut to {int(kw.get('mult', 0.5) * 100)}%", disruptions=["capacity_drop"])
    if kind == "demand_spike":
        dest = kw.get("dest") or dests[int(rng.integers(len(dests)))]
        return Scenario(scenario_id=f"spike-{dest}-{seed}", seed=seed, demand_multiplier={f"{dest}|*": kw.get("mult", 1.4)},
                        description=f"Demand at {dest} up {int((kw.get('mult', 1.4) - 1) * 100)}%", disruptions=["demand_spike"])
    if kind == "new_order":
        dest = kw.get("dest") or dests[int(rng.integers(len(dests)))]
        coms = sorted({c for (d, c) in base.demand if d == dest}) or sorted(base.commodities)
        com = kw.get("commodity") or coms[int(rng.integers(len(coms)))]
        kg = kw.get("kg", 3000.0)
        return Scenario(scenario_id=f"order-{dest}-{com}-{seed}", seed=seed,
                        new_orders=[DemandRow(dest_id=dest, commodity=com, demand_kg=kg)],
                        description=f"New order: {kg:.0f} kg {com} for {dest}", disruptions=["new_order"])
    raise ValueError(f"unknown disruption '{kind}'; choose from {DISRUPTIONS}")


def random_scenario(base: Instance, seed: int, disruption_prob: float = 0.3) -> Scenario:
    """Instance variation for evaluation seeds: season, demand/price swings, inventory, maybe a disruption."""
    rng = np.random.default_rng([seed, 7919])
    dm = {f"{d}|{c}": float(np.exp(rng.normal(0, 0.15))) for (d, c) in base.demand}
    pm = {f"{d}|{c}": float(np.exp(rng.normal(0, 0.10))) for (d, c) in base.prices}
    lm = {b.batch_id: float(np.exp(rng.normal(0, 0.15))) for b in base.batches}
    qm = {b.batch_id: float(np.exp(rng.normal(0, 0.2))) for b in base.batches}
    scen = Scenario(
        scenario_id=f"seed-{seed}", seed=seed, description="random instance variation",
        temp_offset_c=float(rng.normal(0, 2.5)), demand_multiplier=dm, price_multiplier=pm,
        batch_life_multiplier=lm, batch_qty_multiplier=qm,
    )
    if rng.random() < disruption_prob:
        kind = DISRUPTIONS[int(rng.integers(len(DISRUPTIONS)))]
        d = disruption_scenario(base, kind, seed=seed)
        scen = merge_scenarios(scen, d)
    return scen


def merge_scenarios(a: Scenario, b: Scenario) -> Scenario:
    def mul(x, y):
        out = dict(x)
        for k, v in y.items():
            out[k] = out.get(k, 1.0) * v
        return out

    def add(x, y):
        out = dict(x)
        for k, v in y.items():
            out[k] = out.get(k, 0.0) + v
        return out

    return Scenario(
        scenario_id=f"{a.scenario_id}+{b.scenario_id}", seed=a.seed,
        description="; ".join(s for s in (a.description, b.description) if s),
        temp_offset_c=a.temp_offset_c + b.temp_offset_c,
        lane_temp_offset_c=add(a.lane_temp_offset_c, b.lane_temp_offset_c),
        lane_delay_hours=add(a.lane_delay_hours, b.lane_delay_hours),
        capacity_multiplier=mul(a.capacity_multiplier, b.capacity_multiplier),
        demand_multiplier=mul(a.demand_multiplier, b.demand_multiplier),
        price_multiplier=mul(a.price_multiplier, b.price_multiplier),
        new_orders=a.new_orders + b.new_orders,
        batch_life_multiplier=mul(a.batch_life_multiplier, b.batch_life_multiplier),
        batch_qty_multiplier=mul(a.batch_qty_multiplier, b.batch_qty_multiplier),
        disruptions=a.disruptions + b.disruptions,
    )


def seed_range(settings: dict, which: str) -> list[int]:
    lo, hi = settings["scenarios"][f"{which}_seeds"]
    return list(range(int(lo), int(hi)))
