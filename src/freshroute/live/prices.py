"""Price REPLAY (not live) and a simple demand/price forecast baseline.

We do not scrape. If you place Agmarknet-style CSV files in data/prices/ (columns such as
Arrival_Date, Commodity, Market, Modal_Price in INR/quintal), they are replayed. Otherwise a
seeded synthetic seasonal series is used and labeled SYNTHETIC REPLAY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PRICES_DIR

COLUMN_ALIASES = {
    "date": ["Arrival_Date", "arrival_date", "date", "Date", "Price Date"],
    "commodity": ["Commodity", "commodity"],
    "market": ["Market", "market", "Market Name"],
    "modal_price": ["Modal_Price", "Modal_x0020_Price", "modal_price", "Modal Price (Rs./Quintal)", "Modal Price"],
}
BASE = {"tomato": 20.0, "mango": 70.0, "banana": 32.0, "guava": 45.0, "onion": 26.0, "cabbage": 14.0}
PHASE = {"tomato": 200, "mango": 120, "banana": 60, "guava": 300, "onion": 280, "cabbage": 20}


def _pick(df: pd.DataFrame, key: str) -> str | None:
    for c in COLUMN_ALIASES[key]:
        if c in df.columns:
            return c
    return None


def load_replay() -> tuple[pd.DataFrame, str]:
    """Daily price history (date, commodity, market, price_per_kg) and its source label."""
    files = sorted(PRICES_DIR.glob("*.csv")) if PRICES_DIR.exists() else []
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = {k: _pick(df, k) for k in COLUMN_ALIASES}
        if None in cols.values():
            continue
        out = pd.DataFrame({
            "date": pd.to_datetime(df[cols["date"]], dayfirst=True, errors="coerce"),
            "commodity": df[cols["commodity"]].astype(str).str.lower().str.strip(),
            "market": df[cols["market"]].astype(str),
            "price_per_kg": pd.to_numeric(df[cols["modal_price"]], errors="coerce") / 100.0,  # quintal -> kg
        }).dropna()
        frames.append(out)
    if frames:
        return pd.concat(frames, ignore_index=True), "REPLAY (not live) · Agmarknet-style CSV in data/prices"
    return synthetic_history(), "REPLAY (not live) · SYNTHETIC seasonal series"


def synthetic_history(days: int = 730, seed: int = 42, end: str = "2026-10-14") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days, freq="D")
    rows = []
    for c, p0 in BASE.items():
        doy = dates.dayofyear.to_numpy()
        season = 1 + 0.25 * np.sin(2 * np.pi * (doy - PHASE[c]) / 365.0)
        noise = np.exp(np.cumsum(rng.normal(0, 0.02, days)) * 0.3 + rng.normal(0, 0.05, days))
        price = p0 * season * noise
        rows.append(pd.DataFrame({"date": dates, "commodity": c, "market": "synthetic", "price_per_kg": price}))
    return pd.concat(rows, ignore_index=True)


def forecast_next(history: pd.DataFrame, commodity: str, horizon: int = 7) -> pd.DataFrame:
    """Seasonal-naive-with-drift baseline: same day last year scaled by the recent level ratio.

    Returns date, forecast, lo, hi (80% band from historical one-step residual spread).
    """
    h = history[history.commodity == commodity].groupby("date").price_per_kg.mean().sort_index()
    if len(h) < 60:
        raise ValueError(f"not enough history for {commodity}")
    h = h.asfreq("D").interpolate()
    last = h.index[-1]
    recent = h.iloc[-14:].mean()
    out = []
    for k in range(1, horizon + 1):
        d = last + pd.Timedelta(days=k)
        ly = d - pd.Timedelta(days=365)
        if ly in h.index:
            ref = h.loc[ly]
            ref_level = h.loc[ly - pd.Timedelta(days=13): ly].mean() if (ly - pd.Timedelta(days=13)) in h.index else ref
            f = ref * recent / ref_level
        else:
            f = recent
        out.append((d, f))
    resid = (h.pct_change().dropna()).std() * np.sqrt(np.arange(1, horizon + 1))
    df = pd.DataFrame(out, columns=["date", "forecast"])
    df["lo"] = df.forecast * (1 - 1.28 * resid)
    df["hi"] = df.forecast * (1 + 1.28 * resid)
    return df
