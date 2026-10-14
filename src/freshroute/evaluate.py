"""Evaluation harness: tuning set, held-out set, disruption suite, sensitivity.

Seed sets are disjoint (config/settings.yaml):
  tuning   [0, 100)      -> only place where alpha is chosen
  held-out [1000, 1500)  -> headline numbers; nothing is tuned here
  sensitivity [3000, 3060) and disruption suite (held-out seeds 1000-1059 with a forced disruption)
    are reporting-only.

Every method (optimizer, deterministic-optimizer ablation, FIFO, greedy) is planned on the
same scenario instance and simulated on the same realized world.
"""
from __future__ import annotations

import copy
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .adapters import load_base_instance
from .baseline import fifo, greedy_price
from .checker import check_plan
from .montecarlo import compute_risk
from .planner import solve
from .scenarios import DISRUPTIONS, apply_scenario, disruption_scenario, merge_scenarios, random_scenario
from .schema import Instance, Scenario
from .simulator import realize, simulate

METHOD_PLANNER = "FreshRoute planner"
METHOD_DET = "Deterministic optimizer (ablation)"
METHOD_FIFO = "Baseline A: FIFO"
METHOD_GREEDY = "Baseline B: greedy-by-price"
ALL_METHODS = [METHOD_PLANNER, METHOD_DET, METHOD_FIFO, METHOD_GREEDY]
KEY_METRICS = ["spoilage_pct", "fulfillment_pct", "net_revenue", "net_revenue_excl_carry", "spoiled_kg",
               "sold_kg", "diverted_kg", "carried_kg", "unmet_kg", "min_dest_fill_pct", "dest_fill_std_pct",
               "revenue", "transport_cost", "handling_cost", "holding_cost", "spoilage_cost", "salvage_revenue",
               "carry_value", "discount_given"]

_BASE: Instance | None = None


def _init(prefer: str = "auto"):
    global _BASE
    _BASE = load_base_instance(prefer)


def base() -> Instance:
    if _BASE is None:
        _init()
    return _BASE


def deterministic_view(inst: Instance) -> Instance:
    """Same instance with all uncertainty switched off: the optimizer then plans on point estimates."""
    d = copy.copy(inst)
    s = copy.deepcopy(inst.settings)
    s["uncertainty"] = {**s.get("uncertainty", {}), "life_sigma_scale": 0.0, "travel_sigma": 0.0,
                        "delay_prob": 0.0, "temp_sigma0": 0.0, "temp_sigma_per_day": 0.0}
    d.settings = s
    return d


def make_plans(inst: Instance, alpha: float, methods=ALL_METHODS, explain: bool = False):
    risk = compute_risk(inst)
    plans = {}
    if METHOD_PLANNER in methods:
        plans[METHOD_PLANNER] = solve(inst, risk, alpha=alpha, explain=explain)
    if METHOD_DET in methods:
        det = deterministic_view(inst)
        plans[METHOD_DET] = solve(det, compute_risk(det, n=8), alpha=alpha, method=METHOD_DET, explain=False)
    if METHOD_FIFO in methods:
        plans[METHOD_FIFO] = fifo(inst, alpha=alpha)
        plans[METHOD_FIFO].method = METHOD_FIFO
    if METHOD_GREEDY in methods:
        plans[METHOD_GREEDY] = greedy_price(inst, alpha=alpha)
    return risk, plans


def _record(seed, set_name, method, plan, metrics, violations, extra=None):
    rec = {"seed": seed, "set": set_name, "method": method,
           "violations": int(len(violations)),
           "violations_shelf_life": int((violations.constraint == "shelf_life").sum()) if len(violations) else 0,
           "violations_capacity": int(violations.constraint.isin(["trucks", "receiving"]).sum()) if len(violations) else 0,
           "violations_demand": int((violations.constraint == "demand").sum()) if len(violations) else 0,
           "violations_window": int((violations.constraint == "window").sum()) if len(violations) else 0,
           "violations_hold": int((violations.constraint == "hold").sum()) if len(violations) else 0,
           "violations_stock": int((violations.constraint == "stock").sum()) if len(violations) else 0,
           "solve_time_s": float(plan.solve_time_s)}
    for k in KEY_METRICS:
        rec[k] = float(metrics[k])
    for d, v in metrics["unmet_by_dest"].items():
        rec[f"unmet_{d}"] = float(v)
    for d, v in metrics["fill_by_dest"].items():
        rec[f"fill_{d}"] = 100.0 * float(v)
    if extra:
        rec.update(extra)
    return rec


def eval_seed(task) -> list[dict]:
    """task = (seed, set_name, alpha, scenario_override or None)."""
    seed, set_name, alpha, scen_override = task
    b = base()
    scen = scen_override if scen_override is not None else random_scenario(b, seed)
    inst = apply_scenario(b, scen)
    risk, plans = make_plans(inst, alpha)
    world = realize(inst, seed)
    out = []
    for m, p in plans.items():
        v = check_plan(inst, p, risk, alpha)
        out.append(_record(seed, set_name, m, p, simulate(inst, p, world), v,
                           {"disruptions": "|".join(scen.disruptions)}))
    return out


def tune_seed(task) -> list[dict]:
    seed, grid = task
    b = base()
    inst = apply_scenario(b, random_scenario(b, seed))
    risk = compute_risk(inst)
    world = realize(inst, seed)
    out = []
    for a in grid:
        p = solve(inst, risk, alpha=a, explain=False)
        m = simulate(inst, p, world)
        out.append({"seed": seed, "alpha": a, "net_revenue": m["net_revenue"], "spoilage_pct": m["spoilage_pct"],
                    "fulfillment_pct": m["fulfillment_pct"]})
    f = fifo(inst)
    mf = simulate(inst, f, world)
    out.append({"seed": seed, "alpha": "FIFO", "net_revenue": mf["net_revenue"], "spoilage_pct": mf["spoilage_pct"],
                "fulfillment_pct": mf["fulfillment_pct"]})
    return out


def disruption_seed(task) -> list[dict]:
    """Known disruption: replanned optimizer vs. stale plan (made before the disruption) vs. FIFO."""
    seed, kind, alpha = task
    b = base()
    pre_scen = random_scenario(b, seed, disruption_prob=0.0)
    post_scen = merge_scenarios(pre_scen, disruption_scenario(b, kind, seed=seed))
    pre = apply_scenario(b, pre_scen)
    post = apply_scenario(b, post_scen)
    world = realize(post, seed)
    risk_post, plans = make_plans(post, alpha, methods=[METHOD_PLANNER, METHOD_FIFO])
    stale = solve(pre, compute_risk(pre), alpha=alpha, method="Stale plan (no replan)", explain=False)
    plans["Stale plan (no replan)"] = stale
    out = []
    for m, p in plans.items():
        v = check_plan(post, p, risk_post, alpha)
        out.append(_record(seed, f"disruption:{kind}", m, p, simulate(post, p, world), v, {"disruption": kind}))
    return out


SENS_FACTORS = {
    "shelf_life_error": ("life_sigma_scale", [0.5, 1.0, 1.5, 2.0]),
    "price_error": ("price_cv", [0.04, 0.08, 0.16, 0.24]),
    "demand_error": ("demand_cv", [0.05, 0.15, 0.30, 0.45]),
    "delay_probability": ("delay_prob", [0.0, 0.10, 0.25, 0.40]),
}


def sensitivity_seed(task) -> list[dict]:
    """Planner assumptions stay at defaults; only the realized world's noise changes (misspecification test)."""
    seed, alpha, alpha_grid = task
    b = base()
    inst = apply_scenario(b, random_scenario(b, seed))
    risk, plans = make_plans(inst, alpha, methods=[METHOD_PLANNER, METHOD_FIFO])
    out = []
    for factor, (key, levels) in SENS_FACTORS.items():
        for lvl in levels:
            w_inst = copy.copy(inst)
            w_inst.settings = copy.deepcopy(inst.settings)
            w_inst.settings["uncertainty"][key] = lvl
            world = realize(w_inst, seed)
            for m, p in plans.items():
                met = simulate(inst, p, world)
                out.append({"seed": seed, "factor": factor, "level": lvl, "method": m,
                            "net_revenue": met["net_revenue"], "spoilage_pct": met["spoilage_pct"],
                            "fulfillment_pct": met["fulfillment_pct"]})
    world = realize(inst, seed)
    mf = simulate(inst, plans[METHOD_FIFO], world)
    for a in alpha_grid:
        p = solve(inst, risk, alpha=a, explain=False)
        met = simulate(inst, p, world)
        out.append({"seed": seed, "factor": "risk_alpha", "level": a, "method": METHOD_PLANNER,
                    "net_revenue": met["net_revenue"], "spoilage_pct": met["spoilage_pct"],
                    "fulfillment_pct": met["fulfillment_pct"]})
        out.append({"seed": seed, "factor": "risk_alpha", "level": a, "method": METHOD_FIFO,
                    "net_revenue": mf["net_revenue"], "spoilage_pct": mf["spoilage_pct"],
                    "fulfillment_pct": mf["fulfillment_pct"]})
    return out


def run_parallel(fn, tasks, workers: int | None = None, prefer: str = "auto") -> list[dict]:
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    if workers == 1:
        _init(prefer)
        res = [fn(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(prefer,)) as ex:
            res = list(ex.map(fn, tasks, chunksize=max(1, len(tasks) // (workers * 4))))
    return [r for chunk in res for r in chunk]


# ---------------------------------------------------------------- statistics
def summarize(df: pd.DataFrame, metrics=("spoilage_pct", "fulfillment_pct", "net_revenue")) -> dict:
    out = {}
    for m, g in df.groupby("method"):
        out[m] = {}
        for k in metrics:
            x = g[k].to_numpy(float)
            out[m][k] = {"mean": float(np.mean(x)), "median": float(np.median(x)),
                         "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)), "n": int(len(x))}
        out[m]["violations_mean"] = float(g.violations.mean())
        out[m]["plans_with_zero_violations_pct"] = float(100.0 * (g.violations == 0).mean())
        out[m]["solve_time_s_mean"] = float(g.solve_time_s.mean())
        out[m]["solve_time_s_max"] = float(g.solve_time_s.max())
    return out


def paired(df: pd.DataFrame, method: str, baseline: str, metric: str, higher_is_better: bool = True) -> dict:
    a = df[df.method == method].set_index("seed")[metric]
    b = df[df.method == baseline].set_index("seed")[metric]
    d = (a - b.reindex(a.index)).dropna().to_numpy(float)
    n = len(d)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    better = d > 0 if higher_is_better else d < 0
    return {"method": method, "baseline": baseline, "metric": metric, "n": n, "mean_diff": mean,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
            "median_diff": float(np.median(d)), "win_rate_pct": float(100.0 * np.mean(better)),
            "relative_change_pct": float(100.0 * mean / np.mean(b.to_numpy(float))) if np.mean(b) != 0 else float("nan")}
