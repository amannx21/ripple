"""One-command reproduction: ``python -m freshroute.reproduce`` (or ``make reproduce``).

Installs nothing. Loads organizer data if present (else generates the seeded SYNTHETIC
dataset), tunes alpha on the tuning seeds only, evaluates every method on the held-out
seeds, runs the disruption suite and sensitivity analysis, solves the base-instance plan
with decision cards, and writes results/scorecard.json plus figures. All randomness is
seeded; two runs produce identical numbers.

Options: --quick (small seed sets for a smoke run), --workers N, --data auto|synthetic|organizer
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import figures
from .adapters import load_base_instance, organizer
from .checker import check_plan
from .config import FIGURES_DIR, RESULTS_DIR, ROOT, cost_model_source
from .evaluate import (METHOD_FIFO, METHOD_GREEDY, METHOD_PLANNER, METHOD_DET, disruption_seed, eval_seed, paired,
                       run_parallel, sensitivity_seed, summarize, tune_seed)
from .explain import counterfactual_cards
from .montecarlo import compute_risk
from .planner import solve
from .scenarios import DISRUPTIONS, seed_range


def _r(x, nd=4):
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), nd)
    return x


def select_alpha(tune: pd.DataFrame, tol_pct: float) -> tuple[float, pd.DataFrame]:
    """Smallest alpha whose mean tuning-set net revenue is within tol_pct of the best alpha (prefer safety
    when the money difference is negligible). Uses tuning seeds only."""
    g = tune[tune.alpha != "FIFO"].copy()
    g["alpha"] = g.alpha.astype(float)
    g = g.groupby("alpha")[["net_revenue", "spoilage_pct", "fulfillment_pct"]].mean().reset_index()
    best = g.net_revenue.max()
    ok = g[g.net_revenue >= best - abs(best) * tol_pct / 100.0]
    return float(ok.alpha.min()), g


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small seed sets (smoke test); not for headline numbers")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--data", default="auto", choices=["auto", "synthetic", "organizer"])
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    out_dir = RESULTS_DIR / ("quick" if args.quick else "")
    fig_dir = out_dir / "figures" if args.quick else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_instance(args.data)
    s = base.settings
    tuning = seed_range(s, "tuning")
    heldout = seed_range(s, "heldout")
    dis_seeds = seed_range(s, "disruption")
    sens_seeds = seed_range(s, "sensitivity")
    if args.quick:
        tuning, heldout, dis_seeds, sens_seeds = tuning[:12], heldout[:24], dis_seeds[:6], sens_seeds[:6]
    assert not set(tuning) & set(heldout), "tuning and held-out seeds must be disjoint"
    grid = [float(a) for a in s["scenarios"]["alpha_grid"]]
    label = base.source_label
    print(f"[data] {label} dataset: {len(base.batches)} batches, {len(base.destinations)} destinations, {len(base.lanes)} lanes")
    print(f"[cost model] {cost_model_source().relative_to(ROOT)}")

    # 1. tune alpha on the tuning set only
    t0 = time.perf_counter()
    tune = pd.DataFrame(run_parallel(tune_seed, [(sd, grid) for sd in tuning], args.workers, args.data))
    alpha, g = select_alpha(tune, float(s["scenarios"].get("alpha_selection_tolerance_pct", 0.5)))
    tune.to_csv(out_dir / "tuning.csv", index=False)
    print(f"[tuning] seeds {tuning[0]}-{tuning[-1]} ({len(tuning)}): alpha grid {grid} -> chosen alpha {alpha} "
          f"({time.perf_counter() - t0:.0f}s)")

    # 2. held-out evaluation
    t0 = time.perf_counter()
    held = pd.DataFrame(run_parallel(eval_seed, [(sd, "heldout", alpha, None) for sd in heldout], args.workers, args.data))
    held.to_csv(out_dir / "heldout_records.csv", index=False)
    print(f"[held-out] seeds {heldout[0]}-{heldout[-1]} ({len(heldout)}) evaluated ({time.perf_counter() - t0:.0f}s)")

    # 3. organizer hidden scenarios, if any
    org = []
    if label == "ORGANIZER":
        scens = organizer.load_scenarios()
        if scens:
            org = run_parallel(eval_seed, [(sc.seed, f"organizer:{sc.scenario_id}", alpha, sc) for sc in scens],
                               args.workers, args.data)
            pd.DataFrame(org).to_csv(out_dir / "organizer_scenarios.csv", index=False)
            print(f"[organizer] {len(scens)} scenario(s) evaluated")

    # 4. disruption suite
    t0 = time.perf_counter()
    dis = pd.DataFrame(run_parallel(disruption_seed, [(sd, k, alpha) for k in DISRUPTIONS for sd in dis_seeds],
                                    args.workers, args.data))
    dis.to_csv(out_dir / "disruption_records.csv", index=False)
    print(f"[disruptions] {len(DISRUPTIONS)} kinds x {len(dis_seeds)} seeds ({time.perf_counter() - t0:.0f}s)")

    # 5. sensitivity
    t0 = time.perf_counter()
    sens = pd.DataFrame(run_parallel(sensitivity_seed, [(sd, alpha, grid) for sd in sens_seeds], args.workers, args.data))
    sens.to_csv(out_dir / "sensitivity_records.csv", index=False)
    print(f"[sensitivity] seeds {sens_seeds[0]}-{sens_seeds[-1]} ({time.perf_counter() - t0:.0f}s)")

    # 6. base-instance plan with decision cards
    risk = compute_risk(base, keep_samples=False)
    plan = solve(base, risk, alpha=alpha)
    viol = check_plan(base, plan, risk)
    cards = counterfactual_cards(base, risk, plan)
    plan.rows.to_csv(out_dir / "plan.csv", index=False)
    (out_dir / "plan.json").write_text(plan.to_json())
    risk.pairs.to_csv(out_dir / "risk_table.csv", index=False)
    (out_dir / "decision_cards.json").write_text(json.dumps(_r(cards), indent=2))

    # 7. scorecard
    metrics3 = ("spoilage_pct", "fulfillment_pct", "net_revenue")
    comp = []
    for bl in (METHOD_FIFO, METHOD_GREEDY, METHOD_DET):
        comp.append(paired(held, METHOD_PLANNER, bl, "spoilage_pct", higher_is_better=False))
        comp.append(paired(held, METHOD_PLANNER, bl, "fulfillment_pct"))
        comp.append(paired(held, METHOD_PLANNER, bl, "net_revenue"))
    dis_summary = {}
    for k, gk in dis.groupby("disruption"):
        dis_summary[k] = {"summary": summarize(gk, metrics3),
                          "replan_vs_stale_net_revenue": paired(gk, METHOD_PLANNER, "Stale plan (no replan)", "net_revenue"),
                          "planner_vs_fifo_net_revenue": paired(gk, METHOD_PLANNER, METHOD_FIFO, "net_revenue")}
    sens_summary = (sens.groupby(["factor", "level", "method"])[["net_revenue", "spoilage_pct", "fulfillment_pct"]]
                    .mean().reset_index())
    fifo_net = paired(held, METHOD_PLANNER, METHOD_FIFO, "net_revenue")
    cycles = float(s.get("roi", {}).get("dispatch_cycles_per_month", 26))
    total_t = sum(b.quantity_kg for b in base.batches) / 1000.0
    scorecard = {
        "data_source": label,
        "data_note": ("SYNTHETIC DATA: seeded generator, not real-world data; results are not real-world performance"
                      if label == "SYNTHETIC" else "Organizer dataset"),
        "cost_model_source": str(cost_model_source().name) + (" (organizer)" if "organizer" in str(cost_model_source()) else " (bundled placeholder)"),
        "risk_alpha_tuned": alpha,
        "alpha_grid": grid,
        "monte_carlo_samples": int(s.get("monte_carlo_samples", 1000)),
        "seed_sets": {"tuning": [tuning[0], tuning[-1], len(tuning)], "heldout": [heldout[0], heldout[-1], len(heldout)],
                      "disruption": [dis_seeds[0], dis_seeds[-1], len(dis_seeds)],
                      "sensitivity": [sens_seeds[0], sens_seeds[-1], len(sens_seeds)]},
        "quick_mode": bool(args.quick),
        "tuning_table": _r(g.sort_values("alpha").to_dict(orient="records")),
        "alpha_selection_rule": f"smallest alpha within {s['scenarios'].get('alpha_selection_tolerance_pct', 0.5)}% of the best mean tuning-set net revenue",
        "heldout": {"n_scenarios": len(heldout), "summary": _r(summarize(held, metrics3 + ("diverted_kg", "unmet_kg", "min_dest_fill_pct", "net_revenue_excl_carry"))),
                    "paired_vs_baselines": _r(comp)},
        "organizer_scenarios": _r(summarize(pd.DataFrame(org), metrics3)) if org else None,
        "disruptions": _r(dis_summary),
        "sensitivity": _r(sens_summary.to_dict(orient="records")),
        "base_plan": {"status": plan.status, "solve_time_s": round(plan.solve_time_s, 3), "violations": int(len(viol)),
                      "expected_net_inr": round(plan.objective, 1),
                      "kg_by_action": _r(plan.rows.groupby("action").quantity_kg.sum().to_dict(), 1)},
        "roi_estimate": {
            "planner_minus_fifo_net_inr_per_cycle_mean": round(fifo_net["mean_diff"], 1),
            "ci95": [round(fifo_net["ci95_low"], 1), round(fifo_net["ci95_high"], 1)],
            "dispatch_cycles_per_month": cycles,
            "inr_per_month_mean": round(fifo_net["mean_diff"] * cycles, 0),
            "inr_per_month_ci95": [round(fifo_net["ci95_low"] * cycles, 0), round(fifo_net["ci95_high"] * cycles, 0)],
            "stock_per_cycle_tonnes": round(total_t, 1),
            "assumptions": "Held-out mean paired difference vs FIFO per dispatch cycle x cycles per month; "
                           "same cost model, same stock volume; " + ("synthetic data, so this is illustrative only."
                                                                     if label == "SYNTHETIC" else "organizer data."),
        },
    }
    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))

    # 8. figures
    head_label = f"Held-out seeds {heldout[0]}-{heldout[-1]} (n={len(heldout)}), alpha={alpha}, {label} DATA"
    figures.headline(held, fig_dir / "heldout_headline.png", head_label)
    figures.tuning_curve(tune, alpha, fig_dir / "alpha_tuning.png")
    figures.sensitivity(sens, fig_dir / "sensitivity.png")
    figures.disruptions(dis, fig_dir / "disruptions.png")
    (out_dir / "run_log.json").write_text(json.dumps({"runtime_s": round(time.perf_counter() - t_start, 1),
                                                       "finished_unix": int(time.time())}, indent=2))

    sm = scorecard["heldout"]["summary"]
    print("\n=== HEADLINE (held-out, %d scenarios, %s data, alpha=%s) ===" % (len(heldout), label, alpha))
    for m in (METHOD_PLANNER, METHOD_DET, METHOD_FIFO, METHOD_GREEDY):
        x = sm[m]
        print(f"{m:38s} spoilage {x['spoilage_pct']['mean']:6.2f}%  fulfilled {x['fulfillment_pct']['mean']:6.2f}%  "
              f"net INR {x['net_revenue']['mean']:12,.0f}  violations/plan {x['violations_mean']:.2f}")
    for c in comp[:3]:
        print(f"vs FIFO {c['metric']:16s} mean diff {c['mean_diff']:12,.2f}  95% CI [{c['ci95_low']:,.2f}, {c['ci95_high']:,.2f}]  "
              f"better in {c['win_rate_pct']:.0f}% of scenarios")
    print(f"base plan: {plan.status}, {len(viol)} violations, solve {plan.solve_time_s:.2f}s")
    print(f"wrote {out_dir / 'scorecard.json'} in {time.perf_counter() - t_start:.0f}s")
    return scorecard


if __name__ == "__main__":
    main()
