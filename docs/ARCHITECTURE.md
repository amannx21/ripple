# Architecture

```
config/                 settings.yaml, cost_model.yaml (REPLACE WITH ORGANIZER), commodities.yaml
data/organizer/         drop-in for organizer files (+ mapping.yaml, scenarios/, cost_model.yaml)
data/synthetic/         seeded SYNTHETIC dataset (generated on first run)
data/cache/bundled/     cached OSRM + Open-Meteo responses so live mode works offline
data/prices/            optional Agmarknet-style CSVs for price replay
src/freshroute/
  schema.py             pydantic models: Commodity, Hub, Batch, Destination, Lane, PriceRow, DemandRow, CostModel, Scenario; Instance
  config.py             paths; organizer cost model overrides the bundled one
  adapters/             tables.py (read + validate), organizer.py (mapping.yaml), synthetic.py, check.py
  temperature.py        diurnal ambient profiles (offline)
  scenarios.py          apply_scenario, disruption presets, random scenario variation, seed ranges
  shelflife.py          Q10 rate, chilling penalty, consumed life, required life by discount
  montecarlo.py         vectorized SaR table per (batch, destination, discount) + hold viability
  economics.py          expected per-kg values shared by the planner and the explanations
  planner.py            MILP (PuLP → HiGHS): ship/discount/hold/divert, trucks, demand, risk mask, break-even
  checker.py            independent constraint checker (any plan)
  baseline.py           FIFO (A), greedy-by-price (B)
  simulator.py          realize(): one sampled world; simulate(): apply any plan, same accounting for all
  explain.py            reasons, counterfactual decision cards, discount timing
  evaluate.py           tuning / held-out / disruption / sensitivity runners (process pool)
  reproduce.py          one-command pipeline → results/scorecard.json + figures
  figures.py            matplotlib figures
  live/                 cache.py (TTL cache, rate limit, circuit breaker, fallback), providers.py (OSRM, Open-Meteo),
                        __init__.py (live instance), prices.py (replay + forecast baseline)
app/                    Streamlit: Home + 8 pages
tests/                  pytest suite
```

## Data flow

```
organizer files ──► adapters ──► Instance ◄── synthetic generator
                                   │
             scenario seed ──► apply_scenario ──► scenario Instance (what the planner knows)
                                   │
                    ┌──────────────┼────────────────────┐
                    ▼              ▼                    ▼
           Monte Carlo SaR    baselines (FIFO,      deterministic
           (1,000 samples)    greedy)               optimizer (ablation)
                    │              │                    │
                    ▼              │                    │
           MILP planner ───────────┼────────────────────┘
                    │              │
                    ▼              ▼
             independent checker (same α, same risk table)
                    │
     realize(seed) ─┴─► simulator (same world, same cost model) ─► metrics ─► scorecard.json
```

## Why the model stays linear

Arrival-life distributions depend only on batch, lane and departure, not on quantities. The Monte Carlo runs first and gives, for each option, P(ok), P(late) and P(bad). Expected value per kg is then a constant, the risk limit becomes a mask on which options exist, and the MILP only has to handle quantities, trucks, capacity and demand. Each solve takes about 0.1–0.2 s.

## Reproducibility

* All randomness uses `numpy.random.default_rng` with explicit seed tuples: `[seed, stream, index]`.
* The planner's Monte Carlo streams (11/23/31) and the simulator's realization streams (101–105) never overlap.
* The process pool returns results in task order. HiGHS runs single-threaded.
* `scorecard.json` holds no timestamps. Run time is written to `run_log.json`.
