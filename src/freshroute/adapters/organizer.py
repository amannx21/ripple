"""Adapter for the organizers' dataset.

Put the organizer files in data/organizer/. Each table is looked up as
``<table>.csv|json|xlsx`` (batches, destinations, lanes, prices, demand and
optionally hubs, commodities). If file or column names differ, describe the
mapping in data/organizer/mapping.yaml::

    files:
      batches: inventory_batches.xlsx
    columns:
      batches:
        batch_id: BatchID          # our field: their column
        quantity_kg: qty_kg
    defaults:
      lanes:
        ambient_mean_c: 28         # value used when their table has no such column
    unit_scale:
      batches:
        quantity_kg: 1000          # e.g. their quantities are in tonnes

Hidden scenarios go in data/organizer/scenarios/*.yaml|json (Scenario schema).
A cost model at data/organizer/cost_model.yaml overrides config/cost_model.yaml.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from ..config import ORGANIZER_DIR, load_commodities, load_cost_model, load_settings, load_yaml
from ..schema import Commodity, Instance, Scenario
from .tables import OPTIONAL_TABLES, REQUIRED_TABLES, AdapterError, build_instance, find_table_file, read_table


def organizer_data_present(folder: Path = ORGANIZER_DIR) -> bool:
    if not folder.exists():
        return False
    mapping = _load_mapping(folder)
    try:
        return all(find_table_file(folder, t, mapping.get("files", {}).get(t)) for t in REQUIRED_TABLES)
    except AdapterError:
        return True  # files are referenced but broken: surface the error in load()


def _load_mapping(folder: Path) -> dict:
    for name in ("mapping.yaml", "mapping.yml"):
        p = folder / name
        if p.exists():
            return load_yaml(p) or {}
    return {}


def load_tables(folder: Path = ORGANIZER_DIR) -> dict[str, pd.DataFrame]:
    mapping = _load_mapping(folder)
    files = mapping.get("files", {}) or {}
    columns = mapping.get("columns", {}) or {}
    defaults = mapping.get("defaults", {}) or {}
    scales = mapping.get("unit_scale", {}) or {}
    tables = {}
    for t in REQUIRED_TABLES + OPTIONAL_TABLES:
        path = find_table_file(folder, t, files.get(t))
        if path is None:
            if t in REQUIRED_TABLES:
                raise AdapterError(
                    f"[ORGANIZER] required table '{t}' not found in {folder}.\n"
                    f"  Expected a file named {t}.csv / {t}.json / {t}.xlsx, or set files.{t} in mapping.yaml."
                )
            continue
        df = read_table(path)
        rename = {theirs: ours for ours, theirs in (columns.get(t) or {}).items()}
        absent = [theirs for theirs in rename if theirs not in df.columns]
        if absent:
            raise AdapterError(
                f"[ORGANIZER] mapping.yaml maps columns {absent} for table '{t}', but {path.name} has no such columns.\n"
                f"  Columns found: {list(df.columns)}"
            )
        df = df.rename(columns=rename)
        for col, val in (defaults.get(t) or {}).items():
            if col not in df.columns:
                df[col] = val
        for col, factor in (scales.get(t) or {}).items():
            if col in df.columns:
                df[col] = df[col] * float(factor)
        tables[t] = df
    return tables


def load_commodities_override(tables: dict[str, pd.DataFrame]) -> dict[str, Commodity]:
    comms = load_commodities()
    if "commodities" not in tables:
        return comms
    for rec in tables["commodities"].to_dict(orient="records"):
        name = rec.get("name")
        if not name:
            raise AdapterError("[ORGANIZER] commodities table needs a 'name' column.")
        merged = {**(comms[name].model_dump() if name in comms else {}), **{k: v for k, v in rec.items() if pd.notna(v)}}
        try:
            comms[name] = Commodity(**merged)
        except ValidationError as exc:
            raise AdapterError(f"[ORGANIZER] commodity '{name}': {exc.errors()[0]['msg']} ({exc.errors()[0]['loc']})") from exc
    return comms


def load(folder: Path = ORGANIZER_DIR) -> Instance:
    tables = load_tables(folder)
    comms = load_commodities_override(tables)
    missing = sorted(set(tables["batches"]["commodity"]) - set(comms))
    if missing:
        raise AdapterError(
            f"[ORGANIZER] commodities {missing} have no shelf-life parameters.\n"
            "  Add them to config/commodities.yaml or provide a commodities table."
        )
    return build_instance(tables, comms, load_cost_model(), load_settings(), "ORGANIZER")


def load_scenarios(folder: Path = ORGANIZER_DIR) -> list[Scenario]:
    sdir = folder / "scenarios"
    if not sdir.exists():
        return []
    out = []
    for p in sorted(sdir.iterdir()):
        if p.suffix.lower() not in (".yaml", ".yml", ".json"):
            continue
        raw = load_yaml(p)
        items = raw if isinstance(raw, list) else [raw]
        for i, item in enumerate(items):
            item.setdefault("scenario_id", f"{p.stem}-{i}")
            try:
                out.append(Scenario(**item))
            except ValidationError as exc:
                raise AdapterError(f"[ORGANIZER] scenario file {p.name} item {i}: {exc.errors()[0]}") from exc
    return out
