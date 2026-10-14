"""Render results/RESULTS.md from results/scorecard.json and paste it into README.md.

Numbers are copied from the scorecard written by the reproduce run; nothing is typed by hand.
``python -m freshroute.report`` re-renders without re-running the evaluation.
"""
from __future__ import annotations

import json
import re

from .config import RESULTS_DIR, ROOT

ORDER = ["FreshRoute planner", "Deterministic optimizer (ablation)", "Baseline A: FIFO", "Baseline B: greedy-by-price"]


def _inr(x: float) -> str:
    return f"₹{x / 1e5:,.2f} L"


def render(sc: dict) -> str:
    ss = sc["seed_sets"]
    h = sc["heldout"]["summary"]
    lines = [
        f"**{sc['data_source']} DATA** · held-out seeds {ss['heldout'][0]}–{ss['heldout'][1]} "
        f"(**n = {sc['heldout']['n_scenarios']}**) · α = {sc['risk_alpha_tuned']} chosen on tuning seeds "
        f"{ss['tuning'][0]}–{ss['tuning'][1]} only · {sc['monte_carlo_samples']} Monte Carlo samples per pair · "
        f"cost model: {sc['cost_model_source']}",
        "",
        "| Method | Spoilage % mean (p5–p95) | Spoiled + diverted % | Demand fulfilled % mean (p5–p95) | Net revenue per cycle, mean (p5–p95) | Violations per plan |",
        "|---|---|---|---|---|---|",
    ]
    for m in ORDER:
        if m not in h:
            continue
        x = h[m]
        lines.append(
            f"| {m} | {x['spoilage_pct']['mean']:.2f} ({x['spoilage_pct']['p5']:.1f}–{x['spoilage_pct']['p95']:.1f}) "
            f"| {x['spoiled_or_diverted_pct']['mean']:.2f} "
            f"| {x['fulfillment_pct']['mean']:.2f} ({x['fulfillment_pct']['p5']:.1f}–{x['fulfillment_pct']['p95']:.1f}) "
            f"| {_inr(x['net_revenue']['mean'])} ({_inr(x['net_revenue']['p5'])} – {_inr(x['net_revenue']['p95'])}) "
            f"| {x['violations_mean']:.2f} |")
    lines += ["", "Paired difference, FreshRoute minus Baseline A (FIFO), on the same scenarios and realized worlds:", "",
              "| Metric | Mean difference | 95% CI | Planner better in |", "|---|---|---|---|"]
    for c in sc["heldout"]["paired_vs_baselines"]:
        if c["baseline"] != "Baseline A: FIFO":
            continue
        unit = " pts" if c["metric"].endswith("pct") else " INR"
        lines.append(f"| {c['metric']} | {c['mean_diff']:+,.2f}{unit} | [{c['ci95_low']:+,.2f}, {c['ci95_high']:+,.2f}] "
                     f"| {c['win_rate_pct']:.1f}% of scenarios |")
    bp = sc["base_plan"]
    lines += ["", f"Base-instance plan: status {bp['status']}, **{bp['violations']} constraint violations**, solved in "
              f"{bp['solve_time_s']} s. Mean solve time over held-out plans: "
              f"{h['FreshRoute planner']['solve_time_s_mean']:.3f} s (max {h['FreshRoute planner']['solve_time_s_max']:.3f} s)."]
    if sc["data_source"] == "SYNTHETIC":
        lines += ["", "_Synthetic data and placeholder parameters: these numbers describe the method on a generated "
                      "dataset, not real-world performance._"]
    return "\n".join(lines)


def main():
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sc = json.loads((RESULTS_DIR / "scorecard.json").read_text())
    md = render(sc)
    (RESULTS_DIR / "RESULTS.md").write_text("# Results\n\n" + md + "\n", encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text(encoding="utf-8")
        new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
                     lambda _: "<!-- RESULTS:START -->\n" + md + "\n<!-- RESULTS:END -->", txt, flags=re.S)
        readme.write_text(new, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
