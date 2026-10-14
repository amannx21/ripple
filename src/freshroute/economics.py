"""Per-kg economics shared by the planner and the explanations (expected values)."""
from __future__ import annotations

from .schema import Instance, Lane


def transport_var_per_kg(inst: Instance, lane: Lane) -> float:
    t = inst.cost_model.transport
    mult = t.reefer_multiplier if lane.reefer else 1.0
    return t.per_kg_km * lane.distance_km * mult


def full_truck_fixed_per_kg(inst: Instance, lane: Lane) -> float:
    return inst.cost_model.transport.per_truck_fixed / lane.truck_capacity_kg


def salvage_per_kg(inst: Instance, commodity: str) -> float:
    return inst.cost_model.salvage_frac * inst.commodities[commodity].purchase_price_per_kg


def spoil_cost_per_kg(inst: Instance, commodity: str) -> float:
    return inst.commodities[commodity].purchase_price_per_kg


def carry_value_per_kg(inst: Instance, commodity: str) -> float:
    return inst.cost_model.carry_frac * inst.commodities[commodity].purchase_price_per_kg


def ship_value_per_kg(inst: Instance, batch_commodity: str, dest_id: str, lane: Lane, discount: float,
                      p_ok: float, p_late: float, p_bad: float) -> dict:
    """Expected value of shipping one kg on an option, split into parts for explanations."""
    com = inst.commodities[batch_commodity]
    loss = com.handling_loss_frac
    price = inst.price(dest_id, batch_commodity)
    sal = salvage_per_kg(inst, batch_commodity)
    late_value = sal if inst.cost_model.late_delivery == "distress_sale" else 0.0
    late_spoil = 0.0 if inst.cost_model.late_delivery == "distress_sale" else 1.0
    sales = (1 - loss) * price * (1 - discount) * p_ok
    distress = (1 - loss) * late_value * p_late
    spoil = spoil_cost_per_kg(inst, batch_commodity) * ((1 - loss) * (p_bad + late_spoil * p_late) + loss)
    transport = transport_var_per_kg(inst, lane)
    handling = inst.cost_model.handling_cost_per_kg
    net = sales + distress - spoil - transport - handling
    return {"sales": sales, "distress": distress, "spoil": spoil, "transport": transport,
            "handling": handling, "net": net, "price": price}


def hold_value_per_kg(inst: Instance, commodity: str, p_viable: float) -> float:
    days = float(inst.settings.get("planning", {}).get("hold_days", 1))
    return (p_viable * carry_value_per_kg(inst, commodity)
            - (1 - p_viable) * spoil_cost_per_kg(inst, commodity)
            - inst.cost_model.holding_cost_per_kg_day * days)
