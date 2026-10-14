# ripple · FreshRoute Live

Heat-aware, uncertainty-aware dispatch planning for perishable produce. Wild Bugs hackathon, **PS03 FreshRoute**.

> **SYNTHETIC DATA notice.** Until the organizers' dataset is placed in `data/organizer/`, every number in this
> repository comes from a seeded synthetic dataset with placeholder shelf-life and cost parameters.
> These numbers are not real-world performance. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## The idea

Remaining shelf life is not a fixed number. It is a probability distribution, and it shrinks faster in the heat.
FreshRoute:

1. **Samples it.** A vectorized Monte Carlo with 1,000 samples per batch and destination combines shelf-life estimate error, travel time and delays, and temperature forecast error that grows with the horizon.
2. **Prices the risk.** For every batch, destination and discount level it computes **Spoilage-at-Risk (SaR)**: the probability of arriving with less than the minimum sellable life, or too late.
3. **Optimizes under a risk limit α.** A MILP decides **Ship · Discount · Hold · Divert** for every batch, with trucks, capacity, demand, delivery windows, a break-even rule and α. An option is allowed only if P(arrives sellable and on time) ≥ 1 − α.
4. **Explains itself.** Every decision has a plain-language reason. A **counterfactual card** re-solves with that decision forbidden and reports what the next-best alternative would cost.
5. **Is checked independently.** A separate constraint checker validates every plan. The "0 violations" badge appears only when it passes.

## Results (held-out scenarios, computed by `make reproduce`)

<!-- RESULTS:START -->
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

Base-instance plan: status Optimal, **0 constraint violations**, solved in 0.141 s. Mean solve time over held-out plans: 0.290 s (max 0.650 s).

_Synthetic data and placeholder parameters: these numbers describe the method on a generated dataset, not real-world performance._
<!-- RESULTS:END -->

The figures are in `results/figures/` and the full numbers are in `results/scorecard.json`.

## Quick start

Python 3.11+ (developed on 3.12, Windows 11). No API keys and no secrets.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
make reproduce PY=.venv/bin/python   # tune on seeds 0-99, evaluate on 1000-1499, write results/ (~4-5 min on 8 cores)
make test PY=.venv/bin/python        # pytest suite (offline)
make app PY=.venv/bin/python         # Streamlit dashboard at http://localhost:8501
```

Without `make`, use the module commands below. On Windows, `mingw32-make` works with the same targets.

```bash
set PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"   bash: export PYTHONPATH=src
python -m freshroute.reproduce            # add --quick for a 30-second smoke run (not headline numbers)
python -m pytest -q
python -m streamlit run app/Home.py
python -m freshroute.adapters.check       # validate whichever dataset will be used
```

Everything runs offline. Live mode uses bundled cached responses when the network is unavailable.

## Using the organizer dataset

1. Put the files in `data/organizer/`, named `batches`, `destinations`, `lanes`, `prices`, `demand` (and optionally `hubs`, `commodities`) as `.csv`, `.json` or `.xlsx`.
2. If names differ, copy `data/organizer/mapping.example.yaml` to `mapping.yaml`. Map files, columns, default values and unit scales there.
3. Put their cost model in `data/organizer/cost_model.yaml`, using the same keys as `config/cost_model.yaml`. It overrides the placeholder automatically.
4. Put hidden scenarios in `data/organizer/scenarios/*.yaml|json`, following the `Scenario` schema in `src/freshroute/schema.py`. They are evaluated by the same harness with no code changes.
5. Run `python -m freshroute.adapters.check`. A missing field produces a message naming the table, the field, the columns found and how to map it.
6. Run `make reproduce`. Runs are labelled ORGANIZER everywhere.

If the organizers publish a baseline, implement it in `src/freshroute/baseline.py` as the official comparison. FIFO then stays as an extra.

## How scenarios are split

| Set | Seeds | Used for |
|---|---|---|
| Tuning | 0–99 | Choosing α only. Rule: smallest α within 0.5 % of the best mean net revenue |
| Held-out | 1000–1499 | Headline numbers. Nothing is tuned here |
| Disruption suite | 1000–1059 × 5 disruptions | Reporting: replanned vs stale plan vs FIFO |
| Sensitivity | 3000–3059 | Reporting: the world's noise changes while the planner's assumptions stay fixed |

Each seed perturbs the instance: season temperature, demand, prices, life estimates and quantities, plus a random disruption in 30 % of cases. Each seed also samples one realized world from random streams separate from the planner's. All methods are scored on the same scenario and world with the same cost model.

## Methods compared

* **FreshRoute planner.** Monte Carlo SaR plus a risk-limited MILP (PuLP + HiGHS).
* **Deterministic optimizer (ablation).** The same MILP on point estimates. It shows the value of modeling uncertainty.
* **Baseline A: FIFO.** Oldest first, highest-price feasible market, no discount or divert. See [docs/BASELINE.md](docs/BASELINE.md).
* **Baseline B: greedy-by-price.**

## Dashboard pages

1 Scorecard · 2 Plan (dispatch queue, SaR gauges, 0-violations badge, unmet demand, CSV dispatch sheet) ·
3 Live Map & Weather (OSRM routes, Open-Meteo along-route temperature, LIVE/CACHED/FALLBACK badges, cooler-hours
departure scan, simulated sensor, price replay + forecast) · 4 Simulator & Replan (heatwave, delayed truck, capacity
drop, demand spike, new order) · 5 Robustness & Sensitivity · 6 Explain (decision cards, counterfactuals, discount
timing) · 7 ROI · 8 Data & Assumptions.

## Documentation

[ARCHITECTURE](docs/ARCHITECTURE.md) · [ASSUMPTIONS](docs/ASSUMPTIONS.md) · [BASELINE](docs/BASELINE.md) ·
[LIMITATIONS](docs/LIMITATIONS.md) · [DEMO_SCRIPT](docs/DEMO_SCRIPT.md) · JSON schema in `docs/schema.json`.

## Third-party services

* **Open-Meteo** forecast API: free, no key, non-commercial terms. Responses are cached with a 6 h TTL.
* **OSRM public demo server**: light use only. Routes are cached for 30 days. Requests are limited to about 1 per second with a descriptive User-Agent. No live traffic.
* After the first network failure, a 2-minute circuit breaker sends every request to the cached or bundled fallback.
* **Prices are not scraped.** The price module replays Agmarknet-style CSVs placed in `data/prices/`, or uses a synthetic seasonal series, labelled "REPLAY (not live)".

## Open-source libraries

numpy, pandas, scipy (BSD) · pydantic (MIT) · PuLP (BSD) with HiGHS / highspy (MIT) · PyYAML (MIT) · requests
(Apache-2.0) · pytest (MIT) · streamlit (Apache-2.0) · plotly (MIT) · pydeck (Apache-2.0) · matplotlib (PSF-based) ·
openpyxl (MIT). Exact versions are pinned in `requirements.txt`.

## AI assistance

The code in this repository was built with Claude Code (Anthropic) under my direction during the hackathon window.

## License

MIT, see [LICENSE](LICENSE).
