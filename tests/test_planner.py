import math

import pandas as pd
import pytest

from freshroute.baseline import fifo, greedy_price
from freshroute.checker import check_plan
from freshroute.montecarlo import compute_risk
from freshroute.planner import solve
from freshroute.simulator import realize, simulate


@pytest.fixture(scope="module")
def solved(base_instance):
    risk = compute_risk(base_instance, n=600, seed=11)
    plan = solve(base_instance, risk, alpha=0.1)
    return base_instance, risk, plan


def test_planner_optimal_and_zero_violations(solved):
    inst, risk, plan = solved
    assert plan.status == "Optimal"
    assert check_plan(inst, plan, risk).empty
    assert set(plan.rows.action) <= {"SHIP", "DISCOUNT", "HOLD", "DIVERT"}
    assert plan.rows.reason.str.len().min() > 20


def test_stock_conservation(solved):
    inst, _, plan = solved
    for b in inst.batches:
        assert plan.rows[plan.rows.batch_id == b.batch_id].quantity_kg.sum() == pytest.approx(b.quantity_kg, abs=0.5)
    assert (plan.rows.quantity_kg >= 0).all()


def test_capacity_never_exceeded(solved):
    inst, _, plan = solved
    ship = plan.shipments
    for lid, g in ship.groupby("lane_id"):
        lane = inst.lanes[lid]
        assert g.quantity_kg.sum() <= lane.max_trucks * lane.truck_capacity_kg + 1
        assert plan.trucks[lid] <= lane.max_trucks
        assert g.quantity_kg.sum() <= plan.trucks[lid] * lane.truck_capacity_kg + 1
    for d, g in ship.groupby("dest_id"):
        assert g.quantity_kg.sum() <= inst.destinations[d].receiving_capacity_kg + 1


def test_every_shipment_respects_risk_limit(solved):
    _, risk, plan = solved
    pr = risk.pairs.set_index(["batch_id", "dest_id", "discount"])
    for r in plan.shipments.itertuples():
        assert pr.loc[(r.batch_id, r.dest_id, r.discount), "p_ok"] >= 0.9 - 1e-9


@pytest.mark.parametrize("kind", ["stock", "trucks", "demand", "shelf_life", "lane"])
def test_checker_catches_injected_violations(solved, kind):
    inst, risk, plan = solved
    bad = plan.rows.copy()
    if kind == "stock":
        bad.loc[bad.index[0], "quantity_kg"] += 500
    elif kind == "trucks":
        lid = plan.shipments.lane_id.iloc[0]
        lane = inst.lanes[lid]
        j = plan.shipments.index[0]
        bad.loc[j, "quantity_kg"] += lane.max_trucks * lane.truck_capacity_kg
    elif kind == "demand":
        j = plan.shipments.index[0]
        bad.loc[j, "quantity_kg"] += 50000
    elif kind == "shelf_life":
        worst = risk.pairs[(risk.pairs.discount == 0)].sort_values("p_ok").iloc[0]
        bad = pd.concat([bad, pd.DataFrame([{"batch_id": worst.batch_id, "commodity": worst.commodity,
                                             "origin": worst.origin, "action": "SHIP", "dest_id": worst.dest_id,
                                             "lane_id": worst.lane_id, "discount": 0.0, "quantity_kg": 10.0}])],
                        ignore_index=True)
    elif kind == "lane":
        j = plan.shipments.index[0]
        bad.loc[j, "lane_id"] = "NO-SUCH-LANE"
    from copy import copy

    p2 = copy(plan)
    p2.rows = bad
    if kind == "trucks":
        p2.trucks = {}
    v = check_plan(inst, p2, risk)
    expected = {"stock": "stock", "trucks": "trucks", "demand": "demand", "shelf_life": "shelf_life", "lane": "lane"}[kind]
    assert expected in set(v.constraint), v


def test_baselines_conserve_stock_and_respect_capacity(base_instance):
    for p in (fifo(base_instance), fifo(base_instance, by="nearest"), greedy_price(base_instance)):
        for b in base_instance.batches:
            assert p.rows[p.rows.batch_id == b.batch_id].quantity_kg.sum() == pytest.approx(b.quantity_kg, abs=0.5)
        assert (p.rows.discount == 0).all() and not (p.rows.action == "DIVERT").any()
        for lid, g in p.shipments.groupby("lane_id"):
            lane = base_instance.lanes[lid]
            assert g.quantity_kg.sum() <= lane.max_trucks * lane.truck_capacity_kg


def test_fifo_ships_oldest_first(base_instance):
    p = fifo(base_instance)
    ages = {b.batch_id: b.age_days for b in base_instance.batches}
    order = list(dict.fromkeys(p.shipments.batch_id))
    assert [ages[b] for b in order] == sorted([ages[b] for b in order], reverse=True)


def test_simulator_mass_balance_and_determinism(solved):
    inst, _, plan = solved
    w1, w2 = realize(inst, 1234), realize(inst, 1234)
    m1, m2 = simulate(inst, plan, w1), simulate(inst, plan, w2)
    assert m1 == m2
    accounted = m1["sold_kg"] + m1["spoiled_kg"] + m1["diverted_kg"] + m1["carried_kg"] + m1["distress_kg"]
    assert accounted == pytest.approx(m1["total_kg"], rel=1e-9)
    assert 0 <= m1["fulfillment_pct"] <= 100 + 1e-9


def test_planner_and_baseline_scored_on_same_world(solved):
    """The realized world depends only on the instance and seed, never on the plan."""
    inst, _, plan = solved
    w = realize(inst, 77)
    f = fifo(inst)
    a = simulate(inst, plan, w)
    b = simulate(inst, f, w)
    assert a["demand_kg"] == b["demand_kg"] and a["total_kg"] == b["total_kg"]


def test_plan_is_deterministic(base_instance):
    r1 = compute_risk(base_instance, n=500, seed=5)
    r2 = compute_risk(base_instance, n=500, seed=5)
    p1, p2 = solve(base_instance, r1, alpha=0.1), solve(base_instance, r2, alpha=0.1)
    pd.testing.assert_frame_equal(p1.rows, p2.rows)
    assert p1.objective == pytest.approx(p2.objective)


def test_breakeven_rule_blocks_unprofitable_lanes(base_instance):
    """Crash the price at one market: options there must be blocked by the break-even rule, never shipped."""
    import copy

    inst = copy.deepcopy(base_instance)
    inst.prices = {k: (0.5 if k[0] == "MUM" else v) for k, v in inst.prices.items()}
    risk = compute_risk(inst, n=500, seed=2)
    plan = solve(inst, risk, alpha=0.1)
    assert not (plan.shipments.dest_id == "MUM").any()
    assert plan.meta["n_blocked_breakeven"] > 0


def test_heldout_and_tuning_seeds_disjoint(base_instance):
    from freshroute.scenarios import seed_range

    s = base_instance.settings
    assert not set(seed_range(s, "tuning")) & set(seed_range(s, "heldout"))
    assert not set(seed_range(s, "tuning")) & set(seed_range(s, "sensitivity"))
