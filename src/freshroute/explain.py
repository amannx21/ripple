"""Plain-language reasons and counterfactual decision cards."""
from __future__ import annotations

import pandas as pd

from .economics import salvage_per_kg
from .montecarlo import RiskTable
from .schema import Instance


def _inr(x: float) -> str:
    return f"Rs {x:,.1f}"


def _why_not_shipped(inst: Instance, bid: str, opts: pd.DataFrame, alpha: float) -> str:
    o = opts[opts.batch_id == bid]
    if o.empty:
        return "no lane from this hub is available"
    base = o[o.discount == 0]
    best = base.sort_values("p_ok", ascending=False).iloc[0]
    n_allowed = int(o.allowed.sum())
    if n_allowed == 0:
        if not o.risk_ok.any():
            return (f"no destination meets the risk limit: best is {best.dest_id} with "
                    f"P(sellable and on time) {best.p_ok:.0%} < required {1 - alpha:.0%} "
                    f"(median arrival life {best.arr_life_p50:.1f} d vs {best.required_life:.1f} d needed)")
        if not (o.risk_ok & o.breakeven_ok).any():
            r = o[o.risk_ok].sort_values("sales_per_kg", ascending=False).iloc[0]
            return (f"break-even rule: best feasible option ({r.dest_id}) earns {_inr(r.sales_per_kg)}/kg "
                    f"but costs {_inr(r.breakeven_cost_per_kg)}/kg to move")
        return "no destination has demand for this commodity"
    return "the feasible destinations' demand or truck capacity is already used by batches that earn more there"


def attach_reasons(inst: Instance, risk: RiskTable, plan, opts: pd.DataFrame) -> pd.DataFrame:
    df = plan.rows.copy()
    if df.empty:
        df["reason"] = []
        return df
    alpha = plan.alpha
    hold = risk.hold.set_index("batch_id")
    e = inst.cost_model.discount.elasticity
    reasons = []
    for r in df.itertuples():
        com = inst.commodities[r.commodity]
        if r.action in ("SHIP", "DISCOUNT"):
            m = opts[(opts.batch_id == r.batch_id) & (opts.dest_id == r.dest_id) & (opts.discount == r.discount)]
            o = m.iloc[0]
            div = salvage_per_kg(inst, r.commodity)
            s = (f"{r.action.title()} {r.quantity_kg:,.0f} kg to {inst.destinations[r.dest_id].name}: "
                 f"P(arrives sellable and on time) {o.p_ok:.0%} meets the {1 - alpha:.0%} limit; "
                 f"median arrival life {o.arr_life_p50:.1f} d (5th pct {o.arr_life_p5:.1f} d) vs {o.required_life:.1f} d needed "
                 f"at ~{o.transit_temp_mean:.0f} C transit. Expected net {_inr(o.value_per_kg)}/kg vs divert {_inr(div)}/kg.")
            if r.action == "DISCOUNT":
                s += (f" A {r.discount:.0%} markdown lifts demand ~{e * r.discount:.0%} (elasticity assumption) "
                      f"and lowers the life needed to sell from {com.min_sellable_life_days:.1f} d to {o.required_life:.1f} d.")
            reasons.append(s)
        elif r.action == "HOLD":
            pv = float(hold.loc[r.batch_id, "p_viable_after_hold"])
            reasons.append(
                f"Hold {r.quantity_kg:,.0f} kg at the hub: {_why_not_shipped(inst, r.batch_id, opts, alpha)}. "
                f"P(still sellable next cycle) {pv:.0%} (median life after 1 day {hold.loc[r.batch_id, 'life_after_hold_p50']:.1f} d), "
                f"so keeping it is worth more than diverting at {_inr(salvage_per_kg(inst, r.commodity))}/kg.")
        else:
            pv = float(hold.loc[r.batch_id, "p_viable_after_hold"])
            reasons.append(
                f"Divert {r.quantity_kg:,.0f} kg to processing/salvage: {_why_not_shipped(inst, r.batch_id, opts, alpha)}; "
                f"holding is too risky (P(still sellable next cycle) {pv:.0%}). Salvage recovers "
                f"{_inr(salvage_per_kg(inst, r.commodity))}/kg instead of a {_inr(com.purchase_price_per_kg)}/kg spoilage loss.")
    df["reason"] = reasons
    return df


def primary_action(plan, batch_id: str) -> tuple[str, str | None, float]:
    rows = plan.rows[plan.rows.batch_id == batch_id]
    top = rows.sort_values("quantity_kg", ascending=False).iloc[0]
    return top.action, top.dest_id, float(top.quantity_kg)


def counterfactual_cards(inst: Instance, risk: RiskTable, plan, batch_ids: list[str] | None = None) -> list[dict]:
    """For each batch: forbid its main action, re-solve, and report what changes (next-best alternative)."""
    from .planner import solve

    cards = []
    base_obj = plan.objective
    ids = batch_ids or sorted(plan.rows.batch_id.unique())
    for bid in ids:
        action, dest, qty = primary_action(plan, bid)
        if action in ("SHIP", "DISCOUNT"):
            forbid = [(bid, "dest", dest)]
            alt_label = f"not sending it to {dest}"
        elif action == "HOLD":
            forbid = [(bid, "hold", None)]
            alt_label = "not holding it"
        else:
            forbid = [(bid, "divert", None)]
            alt_label = "not diverting it"
        alt = solve(inst, risk, alpha=plan.alpha, forbid=forbid, method="counterfactual")
        alt_rows = alt.rows[alt.rows["batch_id"] == bid] if not alt.rows.empty else alt.rows
        if alt.status != "Optimal":
            alt_desc = (f"no alternative passes the constraints when {alt_label}: every other action breaks the "
                        f"{1 - plan.alpha:.0%} risk limit or a capacity/demand limit (solver status {alt.status})")
            delta = float("nan")
        else:
            alt_desc = "; ".join(
                f"{a.action} {a.quantity_kg:,.0f} kg" + (f" to {a.dest_id}" if isinstance(a.dest_id, str) else "")
                + (f" at {a.discount:.0%} off" if a.discount > 0 else "")
                for a in alt_rows.itertuples())
            delta = alt.objective - base_obj
        b_rows = plan.rows[plan.rows.batch_id == bid]
        ship = b_rows[b_rows.action.isin(["SHIP", "DISCOUNT"])]
        sar_now = float((ship.sar * ship.quantity_kg).sum() / ship.quantity_kg.sum()) if not ship.empty else None
        alt_ship = alt_rows[alt_rows.action.isin(["SHIP", "DISCOUNT"])] if alt.status == "Optimal" else alt_rows.iloc[0:0]
        sar_alt = float((alt_ship.sar * alt_ship.quantity_kg).sum() / alt_ship.quantity_kg.sum()) if not alt_ship.empty else None
        cards.append({
            "batch_id": bid,
            "commodity": inst.batch(bid).commodity,
            "recommendation": "; ".join(
                f"{a.action} {a.quantity_kg:,.0f} kg" + (f" to {a.dest_id}" if isinstance(a.dest_id, str) else "")
                + (f" at {a.discount:.0%} off" if a.discount > 0 else "") for a in b_rows.itertuples()),
            "reasons": list(b_rows.reason),
            "alternative": alt_label,
            "alternative_plan": alt_desc,
            "delta_expected_net_inr": delta,
            "sar_shipped_now": sar_now,
            "sar_shipped_alternative": sar_alt,
        })
    return cards
