"""Hourly ambient temperature profiles along each lane.

Offline mode uses a simple diurnal model (mean + amplitude, peak at 15:00).
Live mode replaces profiles with Open-Meteo forecasts (see freshroute.live).
"""
from __future__ import annotations

import numpy as np


def diurnal_profile(mean_c: float, amp_c: float, hours: int, departure_hour: float, offset_c: float = 0.0) -> np.ndarray:
    clock = (departure_hour + np.arange(hours)) % 24
    return mean_c + offset_c + amp_c * np.cos(2 * np.pi * (clock - 15.0) / 24.0)


def build_forecasts(inst, departure_hour: float | None = None) -> dict[str, np.ndarray]:
    plan = inst.settings.get("planning", {})
    hours = int(plan.get("max_horizon_hours", 96))
    dep = plan.get("departure_hour", 20) if departure_hour is None else departure_hour
    scen = inst.scenario
    out = {}
    for lid, lane in inst.lanes.items():
        off = 0.0
        if scen is not None:
            off = scen.temp_offset_c + scen.lane_temp_offset_c.get(lid, 0.0)
        out[lid] = diurnal_profile(lane.ambient_mean_c, lane.ambient_amp_c, hours, dep, off)
    return out
