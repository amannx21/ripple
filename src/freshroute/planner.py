"""Risk-constrained dispatch optimizer (mixed-integer linear program, PuLP + HiGHS, CBC fallback).

Decision variables
  x[o]   kg shipped on option o = (batch, destination, discount level)
  h[b]   kg held at the hub for the next cycle
  v[b]   kg diverted to processing / salvage
  y[l]   trucks dispatched on lane l (integer)
  u[d,c] unmet base demand (only if the cost model defines a penalty)

Constraints
  stock:      sum_o x[o] + h[b] + v[b] = quantity[b]
  trucks:     sum_{o on l} x[o] <= truck_capacity[l] * y[l],  y[l] <= max_trucks[l]
  receiving:  sum_{o to d} x[o] <= receiving_capacity[d]
  demand:     sum_{o to d, commodity c} x[o] * (1 - loss_c) / (1 + elasticity * k_o) <= demand[d, c]
  shelf life + delivery window: option o exists only if P(ok) >= 1 - alpha  (Monte Carlo mask)
  hold:       h[b] allowed only if P(still sellable next cycle) >= 1 - alpha
  break-even: option o exists only if expected sales revenue per kg covers transport + handling
              per kg at full-truck cost (disclosed rule, see docs/ASSUMPTIONS.md)

Objective: maximize expected net revenue = expected sales + distress-sale salvage + divert salvage
  + carry-over value of held stock - transport (per kg-km and per truck) - handling - holding
  - expected spoilage cost - unmet-demand penalty.
Probabilities are precomputed by Monte Carlo, so the model stays linear.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import warnings

import pulp

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pulp")

from .economics import (full_truck_fixed_per_kg, hold_value_per_kg, salvage_per_kg, ship_value_per_kg,
                        transport_var_per_kg)
from .montecarlo import RiskTable
from .plan import PLAN_COLUMNS, Plan
from .schema import Instance


def _solver(time_limit: int):
    """HiGHS in-process (deterministic, safe in parallel workers); CBC subprocess as fallback."""
    try:
        h = pulp.HiGHS(msg=False, timeLimit=time_limit, threads=1)
        if h.available():
            return h
    except Exception:  # pragma: no cover - depends on installed solvers
        pass
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, threads=1)


def build_options(inst: Instance, risk: RiskTable, alpha: float) -> pd.DataFrame:
    """Every (batch, destination, discount) option with its economics and the reason it is or is not allowed."""
    p = risk.pairs.copy()
    if p.empty:
        return p
    vals = []
    for r in p.itertuples(index=False):
        lane = inst.lanes[r.lane_id]
        v = ship_value_per_kg(inst, r.commodity, r.dest_id, lane, r.discount, r.p_ok, r.p_late, r.p_bad)
        breakeven_cost = v["transport"] + v["handling"] + full_truck_fixed_per_kg(inst, lane)
        vals.append({
            "value_per_kg": v["net"], "sales_per_kg": v["sales"], "price": v["price"],
            "transport_per_kg": v["transport"], "breakeven_cost_per_kg": breakeven_cost,
            "demand_kg": inst.demand_of(r.dest_id, r.commodity),
        })
    p = pd.concat([p.reset_index(drop=True), pd.DataFrame(vals)], axis=1)
    p["risk_ok"] = p.p_ok >= 1.0 - alpha - 1e-12
    p["breakeven_ok"] = p.sales_per_kg >= p.breakeven_cost_per_kg
    p["market_ok"] = (p.price > 0) & (p.demand_kg > 0)
    p["allowed"] = p.risk_ok & p.breakeven_ok & p.market_ok
    return p


def solve(inst: Instance, risk: RiskTable, alpha: float | None = None, forbid: list[tuple] | None = None,
          method: str = "FreshRoute planner", time_limit: int = 30, explain: bool = True) -> Plan:
    """forbid: list of (batch_id, kind, dest_id) with kind in {'dest','ship','hold','divert','discount'}."""
    alpha = float(inst.settings.get("risk_alpha", 0.10) if alpha is None else alpha)
    t0 = time.perf_counter()
    cm = inst.cost_model
    e = cm.discount.elasticity
    opts = build_options(inst, risk, alpha)
    forbid = forbid or []
    allowed = opts[opts.allowed].copy() if not opts.empty else opts
    for (bid, kind, dest) in forbid:
        if allowed.empty:
            break
        if kind == "dest":
            allowed = allowed[~((allowed.batch_id == bid) & (allowed.dest_id == dest))]
        elif kind == "ship":
            allowed = allowed[allowed.batch_id != bid]
        elif kind == "discount":
            allowed = allowed[~((allowed.batch_id == bid) & (allowed.discount > 0))]
    hold_tbl = risk.hold.set_index("batch_id")
    forbid_hold = {b for (b, k, _) in forbid if k == "hold"}
    forbid_div = {b for (b, k, _) in forbid if k == "divert"}

    prob = pulp.LpProblem("freshroute", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", lowBound=0) for i in allowed.index}
    batches = {b.batch_id: b for b in inst.batches}
    h, v = {}, {}
    hold_val, div_val = {}, {}
    for bid, b in batches.items():
        pv = float(hold_tbl.loc[bid, "p_viable_after_hold"]) if bid in hold_tbl.index else 0.0
        hold_ok = pv >= 1.0 - alpha - 1e-12 and bid not in forbid_hold
        h[bid] = pulp.LpVariable(f"h_{bid}", lowBound=0, upBound=b.quantity_kg if hold_ok else 0)
        v[bid] = pulp.LpVariable(f"v_{bid}", lowBound=0, upBound=0 if bid in forbid_div else b.quantity_kg)
        hold_val[bid] = hold_value_per_kg(inst, b.commodity, pv)
        div_val[bid] = salvage_per_kg(inst, b.commodity)
    lanes_used = sorted(set(allowed.lane_id)) if not allowed.empty else []
    y = {l: pulp.LpVariable(f"y_{l.replace('-', '_')}", lowBound=0, upBound=inst.lanes[l].max_trucks, cat="Integer")
         for l in lanes_used}

    obj = []
    for i, r in allowed.iterrows():
        obj.append(r.value_per_kg * x[i])
    for bid in batches:
        obj.append(hold_val[bid] * h[bid] + div_val[bid] * v[bid])
    for l in lanes_used:
        obj.append(-cm.transport.per_truck_fixed * y[l])

    pen = cm.unmet_demand_penalty_per_kg
    u = {}
    for bid, b in batches.items():
        terms = [x[i] for i in allowed.index[allowed.batch_id == bid]]
        prob += pulp.lpSum(terms) + h[bid] + v[bid] == b.quantity_kg, f"stock_{bid}"
    for l in lanes_used:
        idx = allowed.index[allowed.lane_id == l]
        prob += pulp.lpSum(x[i] for i in idx) <= inst.lanes[l].truck_capacity_kg * y[l], f"truck_{l}"
    for d, dest in inst.destinations.items():
        idx = allowed.index[allowed.dest_id == d] if not allowed.empty else []
        if len(idx):
            prob += pulp.lpSum(x[i] for i in idx) <= dest.receiving_capacity_kg, f"recv_{d}"
    for (d, c), D in inst.demand.items():
        idx = allowed.index[(allowed.dest_id == d) & (allowed.commodity == c)] if not allowed.empty else []
        loss = inst.commodities[c].handling_loss_frac
        served = pulp.lpSum(x[i] * (1 - loss) / (1 + e * allowed.at[i, "discount"]) for i in idx)
        if len(idx):
            prob += served <= D, f"demand_{d}_{c}"
        if pen > 0:
            u[(d, c)] = pulp.LpVariable(f"u_{d}_{c}", lowBound=0)
            prob += u[(d, c)] >= D - served, f"unmet_{d}_{c}"
            obj.append(-pen * u[(d, c)])
    prob += pulp.lpSum(obj)
    prob.solve(_solver(time_limit))
    status = pulp.LpStatus[prob.status]
    solve_time = time.perf_counter() - t0

    rows = []
    if status != "Optimal":
        empty = pd.DataFrame(columns=PLAN_COLUMNS)
        return Plan(method=method, rows=empty, trucks={}, alpha=alpha, objective=float("nan"),
                    solve_time_s=solve_time, status=status, meta={"forbid": [list(f) for f in forbid]})
    for i, r in allowed.iterrows():
        q = x[i].value() or 0.0
        if q > 0.5:
            rows.append({
                "batch_id": r.batch_id, "commodity": r.commodity, "origin": r.origin,
                "action": "DISCOUNT" if r.discount > 0 else "SHIP", "dest_id": r.dest_id, "lane_id": r.lane_id,
                "discount": float(r.discount), "quantity_kg": round(q, 1), "sar": float(r.sar), "p_ok": float(r.p_ok),
                "exp_net_inr": round(q * r.value_per_kg, 1),
            })
    for bid, b in batches.items():
        hq, vq = (h[bid].value() or 0.0), (v[bid].value() or 0.0)
        pv = float(hold_tbl.loc[bid, "p_viable_after_hold"]) if bid in hold_tbl.index else 0.0
        if hq > 0.5:
            rows.append({"batch_id": bid, "commodity": b.commodity, "origin": b.origin, "action": "HOLD",
                         "dest_id": None, "lane_id": None, "discount": 0.0, "quantity_kg": round(hq, 1),
                         "sar": 1.0 - pv, "p_ok": pv, "exp_net_inr": round(hq * hold_val[bid], 1)})
        if vq > 0.5:
            rows.append({"batch_id": bid, "commodity": b.commodity, "origin": b.origin, "action": "DIVERT",
                         "dest_id": None, "lane_id": None, "discount": 0.0, "quantity_kg": round(vq, 1),
                         "sar": 0.0, "p_ok": 1.0, "exp_net_inr": round(vq * div_val[bid], 1)})
    df = pd.DataFrame(rows)
    df = _fix_rounding(df, batches)
    trucks = {l: int(round(y[l].value() or 0)) for l in lanes_used if (y[l].value() or 0) > 0.5}
    df = assign_priority(df, risk)
    plan = Plan(method=method, rows=df, trucks=trucks, alpha=alpha,
                objective=float(pulp.value(prob.objective) or 0.0), solve_time_s=solve_time, status=status,
                meta={"n_options": int(len(opts)), "n_allowed": int(len(allowed)),
                      "n_blocked_risk": int((~opts.risk_ok).sum()) if not opts.empty else 0,
                      "n_blocked_breakeven": int((opts.risk_ok & ~opts.breakeven_ok).sum()) if not opts.empty else 0,
                      "forbid": [list(f) for f in forbid]})
    if explain:
        from .explain import attach_reasons  # local import to avoid a cycle

        plan.rows = attach_reasons(inst, risk, plan, opts)
    else:
        plan.rows["reason"] = ""
    return plan


def _fix_rounding(df: pd.DataFrame, batches: dict) -> pd.DataFrame:
    """Rounded quantities must still sum exactly to the batch quantity: push the residue onto the largest row."""
    if df.empty:
        return df
    for bid, b in batches.items():
        m = df.batch_id == bid
        if not m.any():
            continue
        resid = round(b.quantity_kg - df.loc[m, "quantity_kg"].sum(), 1)
        if abs(resid) > 0:
            j = df.loc[m, "quantity_kg"].idxmax()
            df.loc[j, "quantity_kg"] = round(df.loc[j, "quantity_kg"] + resid, 1)
    return df


def assign_priority(df: pd.DataFrame, risk: RiskTable) -> pd.DataFrame:
    """Dispatch queue: batches whose remaining life is smallest relative to their transit time leave first."""
    if df.empty:
        df["priority_rank"] = []
        return df
    ship = df[df.action.isin(["SHIP", "DISCOUNT"])]
    pr = risk.pairs[risk.pairs.discount == 0].set_index(["batch_id", "dest_id"])
    hold = risk.hold.set_index("batch_id")
    score = {}
    for r in ship.itertuples():
        life_h = float(hold.loc[r.batch_id, "life_p50"]) * 24.0
        travel = float(pr.loc[(r.batch_id, r.dest_id), "travel_p95"])
        ratio = life_h / max(travel, 0.1)
        score[r.batch_id] = min(score.get(r.batch_id, math.inf), ratio)
    order = sorted(score, key=lambda b: (score[b], b))
    rank = {b: i + 1 for i, b in enumerate(order)}
    df["priority_rank"] = df.batch_id.map(rank).astype("Int64")
    df["life_transit_ratio"] = df.batch_id.map(score)
    return df.sort_values(["priority_rank", "batch_id", "action"], na_position="last").reset_index(drop=True)
