import numpy as np
import pytest

from freshroute.config import load_commodities
from freshroute.montecarlo import compute_risk
from freshroute.shelflife import arrival_life, consumed_days, rate_multiplier, required_life

COMS = load_commodities()


@pytest.mark.parametrize("name", sorted(COMS))
def test_consumption_monotonic_in_temperature_above_chill(name):
    com = COMS[name]
    lo = (com.chill_threshold_c or 0.0) + 0.5
    temps = np.linspace(lo, 45, 40)
    r = rate_multiplier(temps, com)
    assert np.all(np.diff(r) > 0), "rate must increase with temperature above the chill threshold"


@pytest.mark.parametrize("name", sorted(COMS))
def test_consumption_monotonic_in_time(name):
    com = COMS[name]
    temps = np.full(48, 30.0)
    hours = np.linspace(0, 60, 61)
    c = np.array([consumed_days(temps, h, com)[0] for h in hours])
    assert c[0] == 0
    assert np.all(np.diff(c) > 0)


def test_reference_temperature_consumes_real_time():
    com = COMS["tomato"]
    temps = np.full(48, com.ref_temp_c)
    assert consumed_days(temps, 24.0, com)[0] == pytest.approx(1.0)
    assert arrival_life(5.0, temps, 12.0, com)[0] == pytest.approx(4.5)


def test_q10_doubles_rate_per_10C_when_q10_is_2():
    com = COMS["cabbage"].model_copy(update={"q10": 2.0})
    assert rate_multiplier(com.ref_temp_c + 10, com) == pytest.approx(2.0)


def test_chilling_injury_penalty():
    com = COMS["tomato"]
    just_below = rate_multiplier(com.chill_threshold_c - 0.1, com)
    just_above = rate_multiplier(com.chill_threshold_c + 0.1, com)
    assert just_below > just_above


def test_discount_reduces_required_life():
    com = COMS["mango"]
    assert required_life(com, 0.3, 1.0) < required_life(com, 0.0, 1.0)


def test_monte_carlo_is_deterministic_and_bounded(base_instance):
    a = compute_risk(base_instance, n=500, seed=7)
    b = compute_risk(base_instance, n=500, seed=7)
    assert a.pairs.equals(b.pairs) and a.hold.equals(b.hold)
    p = a.pairs
    assert ((p.p_ok >= 0) & (p.p_ok <= 1)).all()
    assert np.allclose(p.p_ok + p.p_late + p.p_bad, 1.0)
    # more discount -> less required life -> p_ok can only go up
    for _, g in p.groupby(["batch_id", "dest_id"]):
        assert np.all(np.diff(g.sort_values("discount").p_ok.values) >= -1e-12)


def test_heat_raises_spoilage_at_risk(base_instance):
    import copy

    from freshroute.scenarios import apply_scenario, disruption_scenario

    hot = apply_scenario(base_instance, disruption_scenario(base_instance, "heatwave", delta_c=8.0))
    a = compute_risk(base_instance, n=500, seed=3).pairs
    b = compute_risk(hot, n=500, seed=3).pairs
    nonreefer = ~a.lane_id.isin([l for l, x in base_instance.lanes.items() if x.reefer])
    assert (b.sar[nonreefer].values >= a.sar[nonreefer].values - 1e-12).all()
    assert b.sar[nonreefer].sum() > a.sar[nonreefer].sum()
