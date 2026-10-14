"""Polite HTTP access with a JSON cache, TTL, rate limiting and bundled fallbacks.

Order of preference for every request:
  1. fresh cache entry (age < TTL)                          -> source "CACHED"
  2. live HTTP call (rate limited, descriptive User-Agent)   -> source "LIVE"
  3. stale cache entry, then bundled sample in data/cache/bundled/ -> source "FALLBACK"
Callers always get data back or a clear ``LiveUnavailable``; the scored core never uses this module.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from ..config import CACHE_DIR, load_settings

LIVE_DIR = CACHE_DIR / "live"
BUNDLED_DIR = CACHE_DIR / "bundled"
_lock = threading.Lock()
_last_call = [0.0]
_down_until = [0.0]  # circuit breaker: after a network failure, skip live calls for a short while
BREAKER_SECONDS = 120


class LiveUnavailable(RuntimeError):
    pass


@dataclass
class Fetched:
    data: dict
    source: str          # LIVE | CACHED | FALLBACK
    fetched_at: float    # unix time the payload was originally fetched
    note: str = ""

    @property
    def badge(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.fetched_at)) if self.fetched_at else "n/a"
        return f"{self.source} · {ts}"


def cache_key(kind: str, url: str, params: dict) -> str:
    raw = kind + "|" + url + "|" + json.dumps(params, sort_keys=True)
    return f"{kind}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_json(kind: str, url: str, params: dict, offline: bool = False, ttl_hours: float | None = None) -> Fetched:
    cfg = load_settings().get("live", {})
    ttl = float(cfg.get("cache_ttl_hours", 6) if ttl_hours is None else ttl_hours) * 3600
    key = cache_key(kind, url, params)
    live_path = LIVE_DIR / f"{key}.json"
    bundled_path = BUNDLED_DIR / f"{key}.json"
    cached = _read(live_path) if live_path.exists() else None
    if cached and time.time() - cached["fetched_at"] < ttl:
        return Fetched(cached["data"], "CACHED", cached["fetched_at"])
    err = "offline mode"
    if not offline and time.time() < _down_until[0]:
        err = "network recently unreachable (circuit breaker)"
        offline = True
    if not offline:
        try:
            with _lock:
                wait = float(cfg.get("min_seconds_between_requests", 1.1)) - (time.time() - _last_call[0])
                if wait > 0:
                    time.sleep(wait)
                _last_call[0] = time.time()
            r = requests.get(url, params=params, timeout=float(cfg.get("timeout_seconds", 8)),
                             headers={"User-Agent": cfg.get("user_agent", "FreshRoute-demo")})
            r.raise_for_status()
            data = r.json()
            now = time.time()
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            live_path.write_text(json.dumps({"fetched_at": now, "url": url, "params": params, "data": data}), encoding="utf-8")
            return Fetched(data, "LIVE", now)
        except Exception as exc:  # network down, timeout, HTTP error, bad JSON
            err = f"{type(exc).__name__}: {exc}"[:200]
            if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
                _down_until[0] = time.time() + BREAKER_SECONDS
    if cached:
        return Fetched(cached["data"], "FALLBACK", cached["fetched_at"], f"stale cache used ({err})")
    b = _read(bundled_path) if bundled_path.exists() else None
    if b:
        return Fetched(b["data"], "FALLBACK", b["fetched_at"], f"bundled sample used ({err})")
    raise LiveUnavailable(f"{kind}: no live data and no cached sample ({err})")


def bundle_current_cache() -> list[Path]:
    """Copy the live cache into data/cache/bundled/ so the demo runs fully offline."""
    BUNDLED_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(LIVE_DIR.glob("*.json")):
        q = BUNDLED_DIR / p.name
        q.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        out.append(q)
    return out
