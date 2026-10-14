"""Heat-aware shelf-life model.

Remaining life is measured in "days at the commodity's reference temperature".
Spending one hour at temperature T consumes rate(T)/24 days, where

    rate(T) = q10 ** ((T - ref_temp_c) / 10)        (Q10 / Arrhenius-style rule)

multiplied by ``chill_penalty`` below ``chill_threshold_c`` for chill-sensitive
crops. Parameters live in config/commodities.yaml and are PLACEHOLDER ASSUMPTIONS.
"""
from __future__ import annotations

import numpy as np

from .schema import Commodity


def rate_multiplier(temp_c, com: Commodity) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    r = np.power(com.q10, (t - com.ref_temp_c) / 10.0)
    if com.chill_threshold_c is not None and com.chill_penalty > 1.0:
        r = np.where(t < com.chill_threshold_c, r * com.chill_penalty, r)
    return r


def consumed_days(temps: np.ndarray, hours, com: Commodity) -> np.ndarray:
    """Shelf life consumed (days) after ``hours`` hours along hourly ``temps``.

    temps: (H,) or (N, H) hourly temperatures starting at departure.
    hours: scalar or (N,) exposure durations; beyond H the last hour's rate is extended.
    """
    temps = np.atleast_2d(np.asarray(temps, dtype=float))
    hours = np.broadcast_to(np.asarray(hours, dtype=float), (temps.shape[0],))
    if np.any(hours < 0):
        raise ValueError("exposure hours must be non-negative")
    rates = rate_multiplier(temps, com)  # (N, H)
    H = rates.shape[1]
    cum = np.concatenate([np.zeros((rates.shape[0], 1)), np.cumsum(rates, axis=1)], axis=1)  # (N, H+1)
    k = np.minimum(np.floor(hours).astype(int), H)
    idx = np.arange(rates.shape[0])
    base = cum[idx, k]
    last_rate = rates[idx, np.minimum(k, H - 1)]
    frac = hours - k  # fractional hour, or overflow hours when k == H
    return (base + frac * last_rate) / 24.0


def arrival_life(remaining_life_days, temps, hours, com: Commodity) -> np.ndarray:
    return np.asarray(remaining_life_days, dtype=float) - consumed_days(temps, hours, com)


def required_life(com: Commodity, discount: float, sell_speedup: float) -> float:
    """Minimum sellable life on arrival; a markdown sells faster, so it needs less life (assumption)."""
    return com.min_sellable_life_days * max(0.0, 1.0 - sell_speedup * discount)
