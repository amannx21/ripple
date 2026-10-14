"""Seeded, reproducible SYNTHETIC dataset following the FreshRoute schema.

Every file written here carries a ``data_label`` column set to SYNTHETIC and the
folder holds a README saying so. Geography (hub and market coordinates) is real;
distances, travel times, capacities, prices, demand and inventory are synthetic
approximations and must never be presented as real-world performance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import SYNTHETIC_DIR, load_commodities

LABEL = "SYNTHETIC"

HUBS = [
    # hub_id, name, lat, lon, storage_temp_c
    ("NSK", "Nashik aggregation hub (pre-cooled shed)", 19.9975, 73.7898, 18.0),
    ("KLR", "Kolar aggregation hub (open shed)", 13.1367, 78.1292, 27.0),
]

DESTINATIONS = [
    # dest_id, name, lat, lon, receiving_capacity_kg
    ("MUM", "Mumbai (Vashi APMC)", 19.0771, 73.0090, 32000),
    ("PUN", "Pune (Gultekdi market yard)", 18.4900, 73.8640, 22000),
    ("SUR", "Surat (APMC)", 21.1702, 72.8311, 16000),
    ("HYD", "Hyderabad (Bowenpally)", 17.4700, 78.4800, 22000),
    ("CHE", "Chennai (Koyambedu)", 13.0694, 80.1948, 22000),
    ("BLR", "Bengaluru (KR Market)", 12.9640, 77.5770, 26000),
]

LANES = [
    # lane_id, origin, dest, km, hours, truck_kg, max_trucks, reefer, reefer_temp, window_end_h, amb_mean, amb_amp
    ("NSK-MUM", "NSK", "MUM", 167, 4.5, 9000, 3, False, None, 10.0, 28.0, 5.0),
    ("NSK-PUN", "NSK", "PUN", 210, 5.5, 9000, 2, False, None, 10.0, 27.0, 6.0),
    ("NSK-SUR", "NSK", "SUR", 245, 6.0, 9000, 2, False, None, 12.0, 29.0, 5.0),
    ("NSK-HYD", "NSK", "HYD", 620, 13.0, 9000, 2, False, None, 34.0, 30.0, 6.0),
    ("KLR-BLR", "KLR", "BLR", 70, 2.0, 9000, 3, False, None, 10.0, 26.0, 5.0),
    ("KLR-CHE", "KLR", "CHE", 255, 6.0, 9000, 2, False, None, 12.0, 31.0, 4.0),
    ("KLR-HYD", "KLR", "HYD", 570, 11.5, 9000, 2, False, None, 34.0, 30.0, 6.0),
    ("KLR-MUM", "KLR", "MUM", 990, 20.0, 7000, 1, True, 13.0, 34.0, 29.0, 5.0),
]

BASE_PRICE = {"tomato": 20.0, "mango": 70.0, "banana": 32.0, "guava": 45.0, "onion": 26.0, "cabbage": 14.0}
DEST_PRICE_FACTOR = {"MUM": 1.15, "PUN": 1.05, "SUR": 1.10, "HYD": 1.00, "CHE": 1.05, "BLR": 0.95}

BATCH_PLAN = [
    # origin, commodity, count
    ("NSK", "tomato", 4), ("NSK", "onion", 3), ("NSK", "guava", 3), ("NSK", "cabbage", 3),
    ("KLR", "mango", 4), ("KLR", "banana", 4), ("KLR", "tomato", 2), ("KLR", "cabbage", 2),
]


def generate(seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    comms = load_commodities()

    hubs = pd.DataFrame(HUBS, columns=["hub_id", "name", "lat", "lon", "storage_temp_c"])
    dests = pd.DataFrame(DESTINATIONS, columns=["dest_id", "name", "lat", "lon", "receiving_capacity_kg"])
    lanes = pd.DataFrame(LANES, columns=[
        "lane_id", "origin", "dest_id", "distance_km", "base_travel_hours", "truck_capacity_kg", "max_trucks",
        "reefer", "reefer_temp_c", "window_end_hours", "ambient_mean_c", "ambient_amp_c"])

    rows = []
    n = 0
    for origin, com, count in BATCH_PLAN:
        base = comms[com].base_shelf_life_days
        for _ in range(count):
            n += 1
            frac = rng.uniform(0.22, 0.95)
            life = round(base * frac, 2)
            # age is correlated with, but not identical to, consumed life (maturity at harvest varies)
            age = round(max(0.0, base * (1 - frac) * rng.uniform(0.5, 1.3)), 2)
            qty = float(rng.integers(20, 81) * 100)
            rows.append((f"B{n:02d}", com, origin, qty, life, age))
    batches = pd.DataFrame(rows, columns=["batch_id", "commodity", "origin", "quantity_kg", "remaining_life_days", "age_days"])

    reach = {}
    for _, l in lanes.iterrows():
        reach.setdefault(l.origin, set()).add(l.dest_id)
    price_rows, demand_rows = [], []
    for com, base_price in BASE_PRICE.items():
        origins = set(batches.loc[batches.commodity == com, "origin"])
        dests_c = sorted(set().union(*[reach[o] for o in origins]))
        supply = batches.loc[batches.commodity == com, "quantity_kg"].sum()
        ratio = rng.uniform(0.7, 1.15)  # some commodities in glut, some short
        shares = rng.dirichlet(np.ones(len(dests_c)) * 2.0)
        for d, s in zip(dests_c, shares):
            price = base_price * DEST_PRICE_FACTOR[d] * rng.uniform(0.9, 1.1)
            price_rows.append((d, com, round(price, 2)))
            demand_rows.append((d, com, float(round(supply * ratio * s, -1))))
    prices = pd.DataFrame(price_rows, columns=["dest_id", "commodity", "price_per_kg"])
    demand = pd.DataFrame(demand_rows, columns=["dest_id", "commodity", "demand_kg"])

    tables = {"hubs": hubs, "destinations": dests, "lanes": lanes, "batches": batches, "prices": prices, "demand": demand}
    for df in tables.values():
        df["data_label"] = LABEL
    return tables


def write(seed: int = 42, folder: Path = SYNTHETIC_DIR) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in generate(seed).items():
        p = folder / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    readme = folder / "README.md"
    readme.write_text(
        "# SYNTHETIC DATA\n\n"
        f"Generated by `freshroute.adapters.synthetic` with seed {seed}. Hub and market coordinates are real places;\n"
        "distances, travel times, capacities, prices, demand and inventory are synthetic approximations.\n"
        "Nothing in this folder is real-world data. Results computed on it are not real-world performance.\n"
        "Drop organizer files into data/organizer/ to replace it (see data/organizer/README.md).\n",
        encoding="utf-8",
    )
    return paths


def load_tables(folder: Path = SYNTHETIC_DIR, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Read the synthetic CSVs, generating them first if absent."""
    needed = ["hubs", "destinations", "lanes", "batches", "prices", "demand"]
    if not all((folder / f"{t}.csv").exists() for t in needed):
        write(seed, folder)
    return {t: pd.read_csv(folder / f"{t}.csv") for t in needed}


if __name__ == "__main__":
    for p in write():
        print("wrote", p)
