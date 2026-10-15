# Results

**SYNTHETIC DATA** · held-out seeds 1000–1499 (**n = 500**) · α = 0.15 chosen on tuning seeds 0–99 only · 1000 Monte Carlo samples per pair · cost model: cost_model.yaml (bundled placeholder)

| Method | Spoilage % mean (p5–p95) | Spoiled + diverted % | Demand fulfilled % mean (p5–p95) | Net revenue per cycle, mean (p5–p95) | Violations per plan |
|---|---|---|---|---|---|
| FreshRoute planner | 5.31 (2.5–10.7) | 28.11 | 61.87 (49.1–72.1) | ₹26.29 L (₹20.41 L – ₹31.29 L) | 0.00 |
| Deterministic optimizer (ablation) | 18.63 (8.9–29.4) | 26.66 | 58.48 (44.8–70.1) | ₹19.92 L (₹8.44 L – ₹30.35 L) | 17.89 |
| Baseline A: FIFO | 34.37 (23.8–44.7) | 34.37 | 52.01 (39.8–63.9) | ₹10.31 L (₹0.48 L – ₹20.14 L) | 16.94 |
| Baseline B: greedy-by-price | 33.32 (22.7–43.5) | 33.32 | 55.56 (43.1–66.9) | ₹11.77 L (₹2.13 L – ₹21.77 L) | 17.17 |

Paired difference, FreshRoute minus Baseline A (FIFO), on the same scenarios and realized worlds:

| Metric | Mean difference | 95% CI | Planner better in |
|---|---|---|---|
| spoilage_pct | -29.06 pts | [-29.57, -28.55] | 100.0% of scenarios |
| fulfillment_pct | +9.86 pts | [+9.37, +10.35] | 96.2% of scenarios |
| net_revenue | +1,598,290.78 INR | [+1,556,806.18, +1,639,775.38] | 100.0% of scenarios |
| spoiled_or_diverted_pct | -6.26 pts | [-6.69, -5.82] | 91.2% of scenarios |

Base-instance plan: status Optimal, **0 constraint violations**. Planner solve time over held-out plans: mean 0.448 s, max 1.302 s (wall-clock, results/timing.json).

_Synthetic data and placeholder parameters: these numbers describe the method on a generated dataset, not real-world performance._
