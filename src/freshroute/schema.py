"""Data schema for FreshRoute.

Every table the planner consumes is validated against one of these pydantic
models. ``python -m freshroute.schema`` writes the JSON schema to docs/schema.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class Commodity(BaseModel):
    name: str
    base_shelf_life_days: float = Field(gt=0, description="Shelf life at ref_temp_c from fresh (days)")
    ref_temp_c: float = 20.0
    q10: float = Field(ge=1.0, description="Rate multiplier per +10 deg C")
    chill_threshold_c: Optional[float] = Field(None, description="Chilling-injury threshold; None if not chill sensitive")
    chill_penalty: float = Field(1.0, ge=1.0)
    min_sellable_life_days: float = Field(ge=0)
    handling_loss_frac: float = Field(0.0, ge=0, lt=1)
    life_sigma: float = Field(0.2, ge=0, description="Lognormal sigma of the remaining-life estimate")
    purchase_price_per_kg: float = Field(ge=0)


class Hub(BaseModel):
    hub_id: str
    name: str
    lat: float
    lon: float
    storage_temp_c: float = Field(25.0, description="Temperature at which held stock is stored")


class Batch(BaseModel):
    batch_id: str
    commodity: str
    origin: str
    quantity_kg: float = Field(gt=0)
    remaining_life_days: float = Field(ge=0, description="Estimated remaining shelf life at ref temperature")
    age_days: float = Field(0.0, ge=0, description="Days since receipt at hub (used by FIFO)")


class Destination(BaseModel):
    dest_id: str
    name: str
    lat: float
    lon: float
    receiving_capacity_kg: float = Field(gt=0)


class Lane(BaseModel):
    lane_id: str
    origin: str
    dest_id: str
    distance_km: float = Field(gt=0)
    base_travel_hours: float = Field(gt=0)
    truck_capacity_kg: float = Field(gt=0)
    max_trucks: int = Field(ge=0)
    reefer: bool = False
    reefer_temp_c: Optional[float] = None
    window_end_hours: float = Field(gt=0, description="Latest acceptable arrival, hours after departure")
    ambient_mean_c: float = 28.0
    ambient_amp_c: float = 5.0

    @model_validator(mode="after")
    def _reefer_temp(self):
        if self.reefer and self.reefer_temp_c is None:
            raise ValueError(f"lane {self.lane_id}: reefer lane needs reefer_temp_c")
        return self


class PriceRow(BaseModel):
    dest_id: str
    commodity: str
    price_per_kg: float = Field(ge=0)


class DemandRow(BaseModel):
    dest_id: str
    commodity: str
    demand_kg: float = Field(ge=0)


class TransportCost(BaseModel):
    per_kg_km: float = Field(ge=0)
    per_truck_fixed: float = Field(ge=0)
    reefer_multiplier: float = Field(1.0, ge=1.0)


class DiscountRule(BaseModel):
    levels: list[float]
    elasticity: float = Field(ge=0)
    sell_speedup: float = Field(0.0, ge=0)

    @field_validator("levels")
    @classmethod
    def _levels(cls, v):
        v = sorted(float(x) for x in v)
        if not v or v[0] != 0.0:
            raise ValueError("discount levels must include 0.0")
        if any(not (0 <= x < 1) for x in v):
            raise ValueError("discount levels must be in [0, 1)")
        return v


class CostModel(BaseModel):
    currency: str = "INR"
    transport: TransportCost
    handling_cost_per_kg: float = Field(ge=0)
    holding_cost_per_kg_day: float = Field(ge=0)
    discount: DiscountRule
    spoilage: dict = Field(default_factory=lambda: {"cost_basis": "purchase_value"})
    salvage: dict = Field(default_factory=lambda: {"value_fraction_of_purchase": 0.3})
    carry_over: dict = Field(default_factory=lambda: {"value_fraction_of_purchase": 0.8})
    unmet_demand_penalty_per_kg: float = Field(0.0, ge=0)
    late_delivery: Literal["distress_sale", "spoiled"] = "distress_sale"

    @property
    def salvage_frac(self) -> float:
        return float(self.salvage.get("value_fraction_of_purchase", 0.3))

    @property
    def carry_frac(self) -> float:
        return float(self.carry_over.get("value_fraction_of_purchase", 0.8))


class Scenario(BaseModel):
    """A perturbation applied to a base instance (forecast-side changes the planner sees)."""
    scenario_id: str
    seed: int = 0
    description: str = ""
    temp_offset_c: float = 0.0
    lane_temp_offset_c: dict[str, float] = Field(default_factory=dict)
    lane_delay_hours: dict[str, float] = Field(default_factory=dict)
    capacity_multiplier: dict[str, float] = Field(default_factory=dict, description="lane_id or '*' -> multiplier on max_trucks")
    demand_multiplier: dict[str, float] = Field(default_factory=dict, description="keys 'dest|commodity', 'dest|*', '*|commodity' or '*'")
    price_multiplier: dict[str, float] = Field(default_factory=dict, description="same key format as demand_multiplier")
    new_orders: list[DemandRow] = Field(default_factory=list)
    batch_life_multiplier: dict[str, float] = Field(default_factory=dict, description="batch_id or '*' -> multiplier on estimated remaining life")
    batch_qty_multiplier: dict[str, float] = Field(default_factory=dict, description="batch_id or '*' -> multiplier on quantity")
    disruptions: list[str] = Field(default_factory=list, description="human-readable tags, e.g. heatwave")


@dataclass
class Instance:
    """One planning instance: everything the planner sees at decision time."""
    commodities: dict[str, Commodity]
    hubs: dict[str, Hub]
    batches: list[Batch]
    destinations: dict[str, Destination]
    lanes: dict[str, Lane]
    prices: dict[tuple[str, str], float]
    demand: dict[tuple[str, str], float]
    cost_model: CostModel
    settings: dict
    source_label: str = "SYNTHETIC"
    temp_forecast: dict[str, np.ndarray] = field(default_factory=dict)
    temp_source: str = "SYNTHETIC"
    scenario: Optional[Scenario] = None
    notes: list[str] = field(default_factory=list)

    def lane_for(self, origin: str, dest_id: str) -> Optional[Lane]:
        for lane in self.lanes.values():
            if lane.origin == origin and lane.dest_id == dest_id:
                return lane
        return None

    def price(self, dest_id: str, commodity: str) -> float:
        return self.prices.get((dest_id, commodity), 0.0)

    def demand_of(self, dest_id: str, commodity: str) -> float:
        return self.demand.get((dest_id, commodity), 0.0)

    def batch(self, batch_id: str) -> Batch:
        for b in self.batches:
            if b.batch_id == batch_id:
                return b
        raise KeyError(batch_id)

    def validate_references(self) -> list[str]:
        """Cross-table checks. Returns human-readable problems (empty list when clean)."""
        problems = []
        ids = set()
        for b in self.batches:
            if b.batch_id in ids:
                problems.append(f"duplicate batch_id {b.batch_id}")
            ids.add(b.batch_id)
            if b.commodity not in self.commodities:
                problems.append(f"batch {b.batch_id}: unknown commodity '{b.commodity}'")
            if b.origin not in self.hubs:
                problems.append(f"batch {b.batch_id}: unknown origin hub '{b.origin}'")
        for lane in self.lanes.values():
            if lane.origin not in self.hubs:
                problems.append(f"lane {lane.lane_id}: unknown origin '{lane.origin}'")
            if lane.dest_id not in self.destinations:
                problems.append(f"lane {lane.lane_id}: unknown destination '{lane.dest_id}'")
        for (d, c) in list(self.prices) + list(self.demand):
            if d not in self.destinations:
                problems.append(f"price/demand row: unknown destination '{d}'")
            if c not in self.commodities:
                problems.append(f"price/demand row: unknown commodity '{c}'")
        return sorted(set(problems))


SCHEMA_MODELS = [Commodity, Hub, Batch, Destination, Lane, PriceRow, DemandRow, CostModel, Scenario]


def json_schema() -> dict:
    return {m.__name__: m.model_json_schema() for m in SCHEMA_MODELS}


if __name__ == "__main__":
    import json
    from pathlib import Path

    out = Path(__file__).resolve().parents[2] / "docs" / "schema.json"
    out.write_text(json.dumps(json_schema(), indent=2))
    print(f"wrote {out}")
