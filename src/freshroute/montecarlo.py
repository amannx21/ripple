"""Monte Carlo Spoilage-at-Risk (SaR).

For every (batch, lane) pair we sample, vectorized and seeded:
  (a) lognormal noise on the batch's current remaining life,
  (b) travel time: lognormal noise plus an exponential delay with probability p,
  (c) temperature forecast error whose sigma grows with the horizon,
and compute the distribution of remaining life on arrival. Demand and price noise
are sampled by the simulator (they do not change arrival life).

Batches on the same lane share the travel-time and weather draws (they ride the
same trucks on the same road), which keeps comparisons between batches consistent.

Outputs per (batch, destination, discount level):
  p_ok   = P(arrival life >= required life AND arrival within delivery window)
  p_late = P(arrival life >= required life AND arrival after the window)
  p_bad  = P(arrival life <  required life)
  sar    = P(arrival life < minimum sellable life)   <- the headline SaR (undiscounted)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import Instance
from .shelflife import consumed_days, rate_multiplier, required_life


@dataclass
class RiskTable:
    pairs: pd.DataFrame
    hold: pd.DataFrame
    n_samples: int
    seed: int
    samples: dict = field(default_factory=dict)  # (batch_id, dest_id) -> arrival-life samples

    def row(self, batch_id: str, dest_id: str, discount: float = 0.0) -> pd.Series:
        df = self.pairs
        m = (df.batch_id == batch_id) & (df.dest_id == dest_id) & np.isclose(df.discount, discount)
        return df[m].iloc[0]


def _unc(inst: Instance) -> dict:
    return inst.settings.get("uncertainty", {})


def sample_lane(inst: Instance, lane_id: str, n: int, seed: int, lane_idx: int):
    """Travel hours (n,), ambient temperature paths (n, H), and a reefer flag for one lane."""
    u = _unc(inst)
    lane = inst.lanes[lane_id]
    rng = np.random.default_rng([seed, 11, lane_idx])
    s_tt = float(u.get("travel_sigma", 0.12))
    travel = lane.base_travel_hours * np.exp(rng.normal(-0.5 * s_tt**2, s_tt, n))
    delayed = rng.random(n) < float(u.get("delay_prob", 0.1))
    travel = travel + delayed * rng.exponential(float(u.get("delay_hours_mean", 5.0)), n)
    fc = inst.temp_forecast[lane_id]
    H = fc.shape[0]
    sigma_h = float(u.get("temp_sigma0", 1.0)) + float(u.get("temp_sigma_per_day", 1.5)) * np.arange(H) / 24.0
    z = rng.normal(0, 1, (n, 1))
    temps = fc[None, :] + z * sigma_h[None, :]
    return travel, temps


def lane_consumption(inst: Instance, lane_id: str, commodity: str, travel: np.ndarray, temps: np.ndarray) -> np.ndarray:
    lane = inst.lanes[lane_id]
    com = inst.commodities[commodity]
    handling = float(inst.settings.get("planning", {}).get("handling_hours", 3.0))
    if lane.reefer:
        reefer = np.full_like(temps, lane.reefer_temp_c)
        return consumed_days(reefer, travel, com) + consumed_days(temps, np.full_like(travel, handling), com)
    return consumed_days(temps, travel + handling, com)


def sample_batch_life(inst: Instance, batch, n: int, seed: int, batch_idx: int) -> np.ndarray:
    com = inst.commodities[batch.commodity]
    sigma = com.life_sigma * float(_unc(inst).get("life_sigma_scale", 1.0))
    rng = np.random.default_rng([seed, 23, batch_idx])
    return batch.remaining_life_days * np.exp(rng.normal(-0.5 * sigma**2, sigma, n))


def compute_risk(inst: Instance, n: int | None = None, seed: int | None = None, keep_samples: bool = False) -> RiskTable:
    n = int(n or inst.settings.get("monte_carlo_samples", 1000))
    seed = int(inst.settings.get("random_seed", 42) if seed is None else seed)
    cm = inst.cost_model
    levels = cm.discount.levels
    speed = cm.discount.sell_speedup
    hold_days = float(inst.settings.get("planning", {}).get("hold_days", 1))

    lane_ids = sorted(inst.lanes)
    lane_draws = {lid: sample_lane(inst, lid, n, seed, i) for i, lid in enumerate(lane_ids)}
    cons_cache: dict[tuple[str, str], np.ndarray] = {}

    rows, hold_rows, samples = [], [], {}
    for j, b in enumerate(sorted(inst.batches, key=lambda x: x.batch_id)):
        com = inst.commodities[b.commodity]
        L0 = sample_batch_life(inst, b, n, seed, j)
        med_cons = []
        for lid in lane_ids:
            lane = inst.lanes[lid]
            if lane.origin != b.origin or lane.max_trucks <= 0:
                continue
            travel, temps = lane_draws[lid]
            key = (lid, b.commodity)
            if key not in cons_cache:
                cons_cache[key] = lane_consumption(inst, lid, b.commodity, travel, temps)
            cons = cons_cache[key]
            med_cons.append(float(np.median(cons)))
            A = L0 - cons
            late = travel > lane.window_end_hours
            if keep_samples:
                samples[(b.batch_id, lane.dest_id)] = A.astype(np.float32)
            sar = float(np.mean(A < com.min_sellable_life_days))
            q = np.percentile(A, [5, 50, 95])
            tq = np.percentile(travel, [50, 95])
            transit_temp = lane.reefer_temp_c if lane.reefer else float(np.mean(temps[:, : int(lane.base_travel_hours) + 1]))
            for k in levels:
                R = required_life(com, k, speed)
                good = A >= R
                rows.append({
                    "batch_id": b.batch_id, "commodity": b.commodity, "origin": b.origin,
                    "dest_id": lane.dest_id, "lane_id": lid, "discount": k, "required_life": R,
                    "p_ok": float(np.mean(good & ~late)), "p_late": float(np.mean(good & late)),
                    "p_bad": float(np.mean(~good)), "sar": sar, "p_late_any": float(np.mean(late)),
                    "arr_life_p5": q[0], "arr_life_p50": q[1], "arr_life_p95": q[2],
                    "consumed_p50": float(np.median(cons)), "travel_p50": tq[0], "travel_p95": tq[1],
                    "transit_temp_mean": transit_temp,
                })
        hub = inst.hubs[b.origin]
        rng = np.random.default_rng([seed, 31, j])
        store_t = hub.storage_temp_c + rng.normal(0, 1.5, n)
        after = L0 - hold_days * rate_multiplier(store_t, com)
        need = com.min_sellable_life_days + (min(med_cons) if med_cons else 0.0)
        hold_rows.append({
            "batch_id": b.batch_id, "commodity": b.commodity, "origin": b.origin,
            "life_p50": float(np.median(L0)), "life_after_hold_p50": float(np.median(after)),
            "p_viable_after_hold": float(np.mean(after >= need)), "next_cycle_need": need,
        })
    pairs = pd.DataFrame(rows)
    hold = pd.DataFrame(hold_rows)
    return RiskTable(pairs=pairs, hold=hold, n_samples=n, seed=seed, samples=samples)


def feasibility_mask(risk: RiskTable, alpha: float) -> pd.Series:
    """A (batch, destination, discount) option is feasible iff P(ok) >= 1 - alpha."""
    return risk.pairs["p_ok"] >= 1.0 - alpha - 1e-12
