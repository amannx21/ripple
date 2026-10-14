"""Open-Meteo hourly forecasts and OSRM road routes.

Open-Meteo: free, no API key, non-commercial use under their terms; we cache and rate-limit.
OSRM public demo server: light use only; we cache every route and send a descriptive User-Agent.
No live traffic is available from either service.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import load_settings
from .cache import Fetched, get_json


def route(lat1, lon1, lat2, lon2, offline: bool = False) -> tuple[dict, Fetched]:
    cfg = load_settings()["live"]
    url = f"{cfg['osrm_url']}/{lon1:.4f},{lat1:.4f};{lon2:.4f},{lat2:.4f}"
    f = get_json("osrm", url, {"overview": "simplified", "geometries": "geojson"}, offline=offline, ttl_hours=24 * 30)
    r = f.data["routes"][0]
    coords = r["geometry"]["coordinates"]  # [lon, lat]
    return {"distance_km": r["distance"] / 1000.0, "duration_h": r["duration"] / 3600.0,
            "path": [[c[1], c[0]] for c in coords]}, f


def hourly_temperature(points: list[tuple[float, float]], offline: bool = False) -> tuple[list[dict], Fetched]:
    """One Open-Meteo request for several points. Returns [{'time': [...], 'temp': [...]}] per point."""
    cfg = load_settings()["live"]
    params = {"latitude": ",".join(f"{p[0]:.3f}" for p in points),
              "longitude": ",".join(f"{p[1]:.3f}" for p in points),
              "hourly": "temperature_2m", "forecast_days": 4, "timezone": "Asia/Kolkata"}
    f = get_json("openmeteo", cfg["open_meteo_url"], params, offline=offline)
    data = f.data if isinstance(f.data, list) else [f.data]
    return [{"time": d["hourly"]["time"], "temp": d["hourly"]["temperature_2m"]} for d in data], f


def waypoints(path: list[list[float]], n: int) -> list[tuple[float, float]]:
    """n points evenly spaced by distance along a polyline [[lat, lon], ...]."""
    p = np.asarray(path, dtype=float)
    if len(p) == 1:
        return [tuple(p[0])] * n
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]) * np.cos(np.radians(p[:-1, 0])))
    cum = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, cum[-1], n)
    lat = np.interp(targets, cum, p[:, 0])
    lon = np.interp(targets, cum, p[:, 1])
    return [(float(a), float(b)) for a, b in zip(lat, lon)]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
