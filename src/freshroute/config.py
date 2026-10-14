"""Paths and config loading. Organizer-provided files override bundled defaults."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from .schema import Commodity, CostModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
ORGANIZER_DIR = DATA_DIR / "organizer"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
CACHE_DIR = DATA_DIR / "cache"
PRICES_DIR = DATA_DIR / "prices"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() == ".json":
            return json.load(fh)
        return yaml.safe_load(fh) or {}


def load_settings() -> dict:
    return load_yaml(CONFIG_DIR / "settings.yaml")


def cost_model_source() -> Path:
    for name in ("cost_model.yaml", "cost_model.yml", "cost_model.json"):
        p = ORGANIZER_DIR / name
        if p.exists():
            return p
    return CONFIG_DIR / "cost_model.yaml"


def load_cost_model() -> CostModel:
    """Organizer cost model overrides the bundled placeholder (merged key by key)."""
    base = load_yaml(CONFIG_DIR / "cost_model.yaml")
    src = cost_model_source()
    if src.parent == ORGANIZER_DIR:
        base = deep_merge(base, load_yaml(src))
    return CostModel(**base)


def load_commodities() -> dict[str, Commodity]:
    raw = load_yaml(CONFIG_DIR / "commodities.yaml")["commodities"]
    return {name: Commodity(name=name, **vals) for name, vals in raw.items()}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
