"""Shared table reading and instance building used by every adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from ..schema import Batch, Commodity, CostModel, DemandRow, Destination, Hub, Instance, Lane, PriceRow
from ..temperature import build_forecasts

TABLE_MODELS: dict[str, type[BaseModel]] = {
    "hubs": Hub,
    "batches": Batch,
    "destinations": Destination,
    "lanes": Lane,
    "prices": PriceRow,
    "demand": DemandRow,
}
REQUIRED_TABLES = ["batches", "destinations", "lanes", "prices", "demand"]
OPTIONAL_TABLES = ["hubs", "commodities"]
SUPPORTED_EXT = (".csv", ".json", ".xlsx", ".xls")


class AdapterError(ValueError):
    """Raised with a readable explanation when input data cannot be used."""


def find_table_file(folder: Path, table: str, explicit: str | None = None) -> Path | None:
    if explicit:
        p = folder / explicit
        if not p.exists():
            raise AdapterError(f"mapping.yaml points table '{table}' to '{explicit}', but {p} does not exist.")
        return p
    for ext in SUPPORTED_EXT:
        p = folder / f"{table}{ext}"
        if p.exists():
            return p
    return None


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(path, comment="#")
        if ext == ".json":
            return pd.read_json(path)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
    except Exception as exc:  # pragma: no cover - message path
        raise AdapterError(f"Could not read {path.name}: {exc}") from exc
    raise AdapterError(f"Unsupported file type {path.suffix} for {path.name}; use one of {SUPPORTED_EXT}.")


def rows_to_models(df: pd.DataFrame, model: type[BaseModel], table: str, source: str) -> list:
    required = [n for n, f in model.model_fields.items() if f.is_required()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AdapterError(
            f"[{source}] table '{table}' is missing required field(s) {missing}.\n"
            f"  Columns found: {list(df.columns)}\n"
            f"  Fix: rename the columns, or map them in data/organizer/mapping.yaml under columns.{table}."
        )
    known = set(model.model_fields)
    out = []
    for i, rec in enumerate(df.to_dict(orient="records")):
        clean = {k: _clean(v) for k, v in rec.items() if k in known}
        clean = {k: v for k, v in clean.items() if v is not None or not model.model_fields[k].is_required()}
        try:
            out.append(model(**clean))
        except ValidationError as exc:
            errs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise AdapterError(f"[{source}] table '{table}', row {i + 2}: {errs}") from exc
    return out


def _clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def build_instance(
    tables: dict[str, pd.DataFrame],
    commodities: dict[str, Commodity],
    cost_model: CostModel,
    settings: dict,
    source_label: str,
) -> Instance:
    for t in REQUIRED_TABLES:
        if t not in tables:
            raise AdapterError(f"[{source_label}] required table '{t}' not found.")
    batches = rows_to_models(tables["batches"], Batch, "batches", source_label)
    dests = rows_to_models(tables["destinations"], Destination, "destinations", source_label)
    lanes = rows_to_models(tables["lanes"], Lane, "lanes", source_label)
    prices = rows_to_models(tables["prices"], PriceRow, "prices", source_label)
    demand = rows_to_models(tables["demand"], DemandRow, "demand", source_label)
    notes = []
    if "hubs" in tables:
        hubs = rows_to_models(tables["hubs"], Hub, "hubs", source_label)
    else:
        origins = sorted({b.origin for b in batches} | {l.origin for l in lanes})
        hubs = [Hub(hub_id=o, name=o, lat=0.0, lon=0.0) for o in origins]
        notes.append("No hubs table: hubs derived from batch/lane origins with default storage temperature 25 C.")
    inst = Instance(
        commodities=commodities,
        hubs={h.hub_id: h for h in hubs},
        batches=batches,
        destinations={d.dest_id: d for d in dests},
        lanes={l.lane_id: l for l in lanes},
        prices={(p.dest_id, p.commodity): p.price_per_kg for p in prices},
        demand={(r.dest_id, r.commodity): r.demand_kg for r in demand},
        cost_model=cost_model,
        settings=settings,
        source_label=source_label,
        notes=notes,
    )
    problems = inst.validate_references()
    if problems:
        raise AdapterError(f"[{source_label}] cross-table problems:\n  - " + "\n  - ".join(problems))
    inst.temp_forecast = build_forecasts(inst)
    inst.temp_source = "SYNTHETIC (diurnal model)"
    return inst
