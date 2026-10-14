"""Live mode: real road geometry (OSRM) and along-route hourly temperature (Open-Meteo).

This is a second mode on real geography for the dashboard. The scored evaluation never calls it.
Every value carries a source badge: LIVE, CACHED or FALLBACK, with the fetch timestamp.
"""
from __future__ import annotations

import copy
from datetime import datetime

import numpy as np

from ..schema import Instance
from ..temperature import build_forecasts
from .cache import LiveUnavailable
from .providers import hourly_temperature, route, waypoints

TRUCK_SPEED_FACTOR = 1.35  # assumption: loaded trucks take ~35% longer than OSRM's car-profile duration
N_WAYPOINTS = 4


def build_live_instance(base: Instance, departure_hour: int | None = None, offline: bool = False,
                        now: datetime | None = None) -> tuple[Instance, dict]:
    inst = copy.deepcopy(base)
    dep = int(inst.settings.get("planning", {}).get("departure_hour", 20) if departure_hour is None else departure_hour)
    inst.settings = copy.deepcopy(inst.settings)
    inst.settings.setdefault("planning", {})["departure_hour"] = dep
    H = int(inst.settings.get("planning", {}).get("max_horizon_hours", 96))
    status = {"lanes": {}, "weather": None, "notes": []}
    fallback_temps = build_forecasts(inst, dep)

    lanes = {}
    paths = {}
    for lid, lane in inst.lanes.items():
        hub, dest = inst.hubs[lane.origin], inst.destinations[lane.dest_id]
        try:
            r, f = route(hub.lat, hub.lon, dest.lat, dest.lon, offline=offline)
            hours = r["duration_h"] * TRUCK_SPEED_FACTOR
            lanes[lid] = lane.model_copy(update={"distance_km": round(r["distance_km"], 1),
                                                 "base_travel_hours": round(hours, 2)})
            paths[lid] = r["path"]
            status["lanes"][lid] = {"route_source": f.badge, "distance_km": r["distance_km"], "travel_h": hours,
                                    "note": f.note}
        except (LiveUnavailable, KeyError, IndexError) as exc:
            lanes[lid] = lane
            paths[lid] = [[hub.lat, hub.lon], [dest.lat, dest.lon]]
            status["lanes"][lid] = {"route_source": "FALLBACK · synthetic lane", "distance_km": lane.distance_km,
                                    "travel_h": lane.base_travel_hours, "note": str(exc)[:160]}
    inst.lanes = lanes

    pts, owner = [], []
    for lid in sorted(paths):
        for w in waypoints(paths[lid], N_WAYPOINTS):
            pts.append(w)
            owner.append(lid)
    temps = dict(fallback_temps)
    try:
        series, f = hourly_temperature(pts, offline=offline)
        status["weather"] = f.badge
        if f.note:
            status["notes"].append(f.note)
        now = now or datetime.now()
        times = [datetime.fromisoformat(t) for t in series[0]["time"]]
        start = _departure_index(times, now, dep)
        for lid in sorted(paths):
            idx = [i for i, o in enumerate(owner) if o == lid]
            grid = np.array([_extend(np.asarray(series[i]["temp"], dtype=float), start, H) for i in idx])  # (W, H)
            travel = inst.lanes[lid].base_travel_hours
            frac = np.clip(np.arange(H) / max(travel, 1e-6), 0, 1) * (len(idx) - 1)
            lo = np.floor(frac).astype(int)
            hi = np.minimum(lo + 1, len(idx) - 1)
            wgt = frac - lo
            temps[lid] = grid[lo, np.arange(H)] * (1 - wgt) + grid[hi, np.arange(H)] * wgt
            status["lanes"][lid]["temp_source"] = f.badge
            status["lanes"][lid]["temp_max_on_trip"] = float(np.max(temps[lid][: int(np.ceil(travel)) + 1]))
    except (LiveUnavailable, KeyError, IndexError, ValueError) as exc:
        status["weather"] = "FALLBACK · synthetic diurnal model"
        status["notes"].append(f"weather unavailable: {str(exc)[:160]}")
        for lid in paths:
            status["lanes"][lid]["temp_source"] = status["weather"]
    inst.temp_forecast = temps
    inst.temp_source = status["weather"]
    status["paths"] = paths
    status["departure_hour"] = dep
    inst.notes = inst.notes + ["LIVE MODE: real road distances/durations (OSRM, no live traffic) and Open-Meteo "
                               "forecast temperatures; prices are REPLAY (not live)."]
    return inst, status


def _departure_index(times, now: datetime, dep_hour: int) -> int:
    for i, t in enumerate(times):
        if t >= now.replace(minute=0, second=0, microsecond=0) and t.hour == dep_hour:
            return i
    for i, t in enumerate(times):  # forecast is older than today: use the first matching hour
        if t.hour == dep_hour:
            return i
    return 0


def _extend(series: np.ndarray, start: int, H: int) -> np.ndarray:
    """H hours from ``start``; beyond the forecast end, repeat the last full day (persistence)."""
    s = list(series[start:])
    while len(s) < H:
        s.extend(series[-24:] if len(series) >= 24 else series)
    return np.asarray(s[:H], dtype=float)
