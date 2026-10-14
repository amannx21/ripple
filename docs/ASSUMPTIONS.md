# Assumptions

Every number below is a **placeholder assumption** chosen to be plausible. None is a measured or validated constant.
Organizer-provided data and cost models replace them automatically (see README, "Using the organizer dataset").

## Data

| Item | Assumption |
|---|---|
| Dataset | Until organizer files exist in `data/organizer/`, a seeded **SYNTHETIC** dataset is generated (`data/synthetic/`, seed 42). Hub and market coordinates are real places (Nashik, Kolar, Mumbai Vashi APMC, Pune, Surat, Hyderabad, Chennai Koyambedu, Bengaluru KR Market). Inventory, demand, prices, capacities and lane parameters are synthetic. |
| Batches | 25 batches of 6 commodities at 2 hubs. Remaining life = base shelf life × U(0.22, 0.95). Age (used by FIFO) is correlated with, but not equal to, life consumed, because harvest maturity varies. |
| Demand | Per commodity, total demand = supply × U(0.70, 1.15), split over reachable markets by a Dirichlet draw. Only markets reachable by a lane get demand. |
| Prices | Base wholesale price per commodity × market factor × U(0.9, 1.1). |
| Lanes | Distances and times approximate the real corridors. Live mode replaces them with OSRM road routes. |
| Brief layout | The brief mentions `data/adapters/organizer.py`; the adapter lives in `src/freshroute/adapters/organizer.py` (the package layout in section 10). It reads files from `data/organizer/`. |

## Shelf-life model (`config/commodities.yaml`)

* Remaining life is measured in days at the commodity's reference temperature (20 °C; 25 °C for onion).
* One hour at temperature T consumes `rate(T)/24` days, with `rate(T) = Q10^((T − Tref)/10)`.
* Chill-sensitive crops (tomato < 12.5 °C, mango < 12 °C, banana < 13 °C, guava < 8 °C) have their rate multiplied by a chill penalty below the threshold.
* Arrival life = life at dispatch − life consumed over travel time plus 3 handling hours. Reefer lanes run at their set point during travel; handling is at ambient.
* Minimum sellable life on arrival is per commodity (e.g. tomato 2 d, mango 3 d, onion 7 d). A markdown of `k` lowers it to `min × (1 − sell_speedup × k)` because discounted stock sells faster.
* Handling loss (2–5 %) of every shipped kg is lost.
* The parameter values are rough defaults meant for calibration against published shelf-life ranges, for example ranges reported in reviews of the Indian tomato supply chain. They are not validated. The MoFPI/NABCONS post-harvest loss study (2020–22) gives category-level loss averages only (fruits about 6.02–15.05 %, vegetables about 4.87–11.61 %), not batch-level shelf life, which is why shelf life is treated as uncertain.

## Uncertainty (`config/settings.yaml → uncertainty`)

| Source | Model |
|---|---|
| Current remaining life | Lognormal, mean-preserving, sigma per commodity (0.15–0.25) × `life_sigma_scale` |
| Travel time | Base time × lognormal(sigma 0.12), plus with probability 0.10 an exponential delay (mean 5 h) |
| Temperature forecast | Forecast + z × (1.0 + 1.5 × hours/24) °C, one z per sample (error grows with horizon) |
| Demand, price | Realized = forecast × lognormal (CV 0.15 demand, 0.08 price); sampled by the simulator |
| Hub storage temperature | Storage set point ± N(0, 1.5 °C) |

Monte Carlo uses 1,000 samples per (batch, destination) pair, vectorized with numpy and seeded. Batches on the same lane share travel-time and weather draws (common random numbers), because they ride the same trucks.

## Planner

* Risk limit α: an option (batch, destination, discount) is allowed only if P(arrives with enough life AND within the delivery window) ≥ 1 − α. α is tuned on tuning seeds 0–99 only: the **smallest α whose mean tuning-set net revenue is within 0.5 % of the best** (prefer safety when the money difference is negligible).
* Break-even rule: an option is excluded when expected sales revenue per kg is below transport + handling per kg at full-truck cost.
* Hold is allowed only if P(still sellable next cycle) ≥ 1 − α. "Sellable next cycle" means life after one day at hub storage temperature still covers the shortest lane's consumption plus the minimum sellable life.
* Demand is capped on shipped sellable kg, not on expected sold kg, so the plan never knowingly over-ships. A kg sold at discount k uses 1/(1 + elasticity × k) of base demand (linear elasticity 1.5).
* Per-truck fixed cost makes the model a small MILP with integer trucks per lane. It is solved with HiGHS through PuLP (CBC fallback).

## Cost model (`config/cost_model.yaml`, replace with the organizer's)

| Item | Value |
|---|---|
| Transport | ₹0.004 per kg-km + ₹2,500 per truck; reefer × 1.6 |
| Handling | ₹0.8 per kg shipped |
| Holding | ₹0.3 per kg per day |
| Spoilage cost | purchase value of each spoiled kg |
| Salvage (divert) | 30 % of purchase value |
| Carry-over | Stock still sellable next cycle (held, or unsold at a market with ≥ 1 day of margin) is valued at 80 % of purchase value. Reported separately; `net_revenue_excl_carry` is also reported. |
| Late delivery | Rejected and sold at salvage value (distress sale) |
| Unmet-demand penalty | 0 (no penalty unless the organizer defines one) |

## Simulator accounting (identical for every method)

Handling loss is spoiled. Arrival with life below the required life is spoiled. A late arrival is a distress sale. Sellable stock is sold against realized demand, deepest discount first, then shortest life first. Unsold stock at a market is carried over if it has at least 1 day more than the minimum sellable life, otherwise spoiled. Held stock is carried over if still sellable next cycle, otherwise spoiled. Diverted stock earns salvage.

Spoilage % = spoiled kg / total stock kg. Fulfillment % = base-demand units served / realized demand. Diverted kg is reported separately, and `spoiled_or_diverted_pct` is reported so that diverting cannot hide loss.

## Evaluation

* Seeds: tuning 0–99, held-out 1000–1499, sensitivity 3000–3059, disruption suite = held-out seeds 1000–1059 with a forced, known disruption. α is chosen on tuning seeds only.
* Each scenario seed varies the instance (season temperature offset ±2.5 °C, demand ±15 %, price ±10 %, life estimates ±15 %, quantities ±20 %), and 30 % of scenarios add a random disruption.
* The realized world uses separate random streams from the planner's Monte Carlo, so the planner never sees the realization.
* The deterministic optimizer (same MILP with all uncertainty switched off) is reported as an ablation, to isolate the value of modeling uncertainty.

## Live mode

* OSRM car-profile durations × 1.35 for loaded trucks (assumption). No live traffic.
* Open-Meteo hourly 2 m temperature at 4 waypoints per route. The along-route temperature for hour h is interpolated between waypoints by trip progress. Beyond the 4-day forecast, the last day is repeated.
* Prices are REPLAY (not live): Agmarknet-style CSV in `data/prices/` if present, else a seeded synthetic seasonal series. The 7-day forecast is seasonal-naive with level adjustment.

## ROI

The ROI page multiplies the held-out mean paired net-revenue difference against FIFO by dispatch cycles per month (26 by default). It scales linearly by stock volume. On synthetic data this is illustrative only.
