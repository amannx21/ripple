"""Baseline strategies (documented precisely in docs/BASELINE.md).

Baseline A, FIFO (the official comparison unless organizers provide one):
  1. Sort batches oldest first (age_days descending, then batch_id).
  2. For each batch, rank its reachable destinations by price for that commodity, highest first
     (variant "nearest": by distance).
  3. Point-estimate feasibility only: ship to a destination if
     estimated remaining life - (base travel + handling hours)/24 >= minimum sellable life
     (shelf life consumed at the reference temperature, i.e. no heat or uncertainty) and the base
     travel time is inside the delivery window.
  4. Ship min(batch left, remaining demand / (1 - handling loss), lane truck capacity left,
     destination receiving capacity left). Move to the next destination until the batch is gone
     or no destination has room.
  5. No discounting and no diverting. Whatever is not shipped is HELD at the hub.
Baseline B, greedy-by-price: same rules, but processes (batch, destination) pairs in order of
  price descending (ties: oldest batch first).
"""
from __future__ import annotations

import math

import pandas as pd

from .plan import Plan
from .schema import Instance


def _point_feasible(inst: Instance, b, lane) -> bool:
    com = inst.commodities[b.commodity]
    handling = float(inst.settings.get("planning", {}).get("handling_hours", 3.0))
    life_on_arrival = b.remaining_life_days - (lane.base_travel_hours + handling) / 24.0
    return life_on_arrival >= com.min_sellable_life_days and lane.base_travel_hours <= lane.window_end_hours


def _state(inst: Instance):
    lane_left = {l: lane.max_trucks * lane.truck_capacity_kg for l, lane in inst.lanes.items()}
    recv_left = {d: dest.receiving_capacity_kg for d, dest in inst.destinations.items()}
    dem_left = dict(inst.demand)
    return lane_left, recv_left, dem_left


def _ship(inst, b, lane, left, lane_left, recv_left, dem_left):
    loss = inst.commodities[b.commodity].handling_loss_frac
    dem_kg = dem_left.get((lane.dest_id, b.commodity), 0.0) / (1 - loss)
    q = min(left, dem_kg, lane_left[lane.lane_id], recv_left[lane.dest_id])
    q = math.floor(q)
    if q <= 0:
        return 0.0
    lane_left[lane.lane_id] -= q
    recv_left[lane.dest_id] -= q
    dem_left[(lane.dest_id, b.commodity)] -= q * (1 - loss)
    return float(q)


def _finish(inst: Instance, rows: list, method: str, alpha: float) -> Plan:
    df = pd.DataFrame(rows)
    ship = df[df.action == "SHIP"]
    trucks = {l: int(math.ceil(g.quantity_kg.sum() / inst.lanes[l].truck_capacity_kg - 1e-9))
              for l, g in ship.groupby("lane_id")}
    order = list(dict.fromkeys(ship.batch_id))
    rank = {b: i + 1 for i, b in enumerate(order)}
    df["priority_rank"] = df.batch_id.map(rank).astype("Int64")
    df["sar"] = float("nan")
    df["p_ok"] = float("nan")
    df["exp_net_inr"] = float("nan")
    return Plan(method=method, rows=df, trucks=trucks, alpha=alpha, status="heuristic")


def fifo(inst: Instance, by: str = "price", alpha: float | None = None) -> Plan:
    alpha = float(inst.settings.get("risk_alpha", 0.10) if alpha is None else alpha)
    lane_left, recv_left, dem_left = _state(inst)
    rows = []
    for b in sorted(inst.batches, key=lambda x: (-x.age_days, x.batch_id)):
        lanes = [l for l in inst.lanes.values() if l.origin == b.origin and l.max_trucks > 0]
        if by == "price":
            lanes.sort(key=lambda l: (-inst.price(l.dest_id, b.commodity), l.distance_km))
        else:
            lanes.sort(key=lambda l: (l.distance_km, -inst.price(l.dest_id, b.commodity)))
        left = b.quantity_kg
        for lane in lanes:
            if left <= 0:
                break
            if inst.price(lane.dest_id, b.commodity) <= 0 or not _point_feasible(inst, b, lane):
                continue
            q = _ship(inst, b, lane, left, lane_left, recv_left, dem_left)
            if q > 0:
                rows.append(_row(b, "SHIP", lane, q, "FIFO: oldest batch first, highest-price feasible market"))
                left -= q
        if left > 0.5:
            rows.append(_row(b, "HOLD", None, left, "FIFO: no destination with remaining demand/capacity"))
    return _finish(inst, rows, f"Baseline A: FIFO ({by})", alpha)


def greedy_price(inst: Instance, alpha: float | None = None) -> Plan:
    alpha = float(inst.settings.get("risk_alpha", 0.10) if alpha is None else alpha)
    lane_left, recv_left, dem_left = _state(inst)
    left = {b.batch_id: b.quantity_kg for b in inst.batches}
    pairs = []
    for b in inst.batches:
        for lane in inst.lanes.values():
            if lane.origin == b.origin and lane.max_trucks > 0 and inst.price(lane.dest_id, b.commodity) > 0:
                pairs.append((b, lane))
    pairs.sort(key=lambda bl: (-inst.price(bl[1].dest_id, bl[0].commodity), -bl[0].age_days, bl[0].batch_id))
    rows = []
    for b, lane in pairs:
        if left[b.batch_id] <= 0 or not _point_feasible(inst, b, lane):
            continue
        q = _ship(inst, b, lane, left[b.batch_id], lane_left, recv_left, dem_left)
        if q > 0:
            rows.append(_row(b, "SHIP", lane, q, "Greedy: highest price first"))
            left[b.batch_id] -= q
    for b in inst.batches:
        if left[b.batch_id] > 0.5:
            rows.append(_row(b, "HOLD", None, left[b.batch_id], "Greedy: unallocated"))
    return _finish(inst, rows, "Baseline B: greedy-by-price", alpha)


def _row(b, action, lane, q, reason):
    return {"batch_id": b.batch_id, "commodity": b.commodity, "origin": b.origin, "action": action,
            "dest_id": lane.dest_id if lane else None, "lane_id": lane.lane_id if lane else None,
            "discount": 0.0, "quantity_kg": float(q), "reason": reason}
