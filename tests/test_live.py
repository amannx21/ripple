"""Live layer must never break: with the network disabled it falls back to bundled samples, then to synthetic."""
import numpy as np
import pytest
import requests

from freshroute.live import build_live_instance, cache
from freshroute.live.prices import forecast_next, load_replay


@pytest.fixture
def no_network(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise requests.ConnectionError("network disabled in test")

    monkeypatch.setattr(cache.requests, "get", boom)
    monkeypatch.setattr(cache, "LIVE_DIR", tmp_path / "live")  # no fresh cache available
    monkeypatch.setattr(cache, "_down_until", [0.0])
    return tmp_path


def test_fallback_to_bundled_samples(no_network, base_instance):
    if not any(cache.BUNDLED_DIR.glob("*.json")):
        pytest.skip("no bundled samples in data/cache/bundled")
    inst, status = build_live_instance(base_instance, departure_hour=20)
    assert status["weather"].startswith("FALLBACK")
    assert all(s["route_source"].startswith("FALLBACK") for s in status["lanes"].values())
    for lid, t in inst.temp_forecast.items():
        assert t.shape == (96,) and np.isfinite(t).all()


def test_fallback_to_synthetic_when_nothing_cached(no_network, monkeypatch, base_instance):
    monkeypatch.setattr(cache, "BUNDLED_DIR", no_network / "empty")
    inst, status = build_live_instance(base_instance, departure_hour=20)
    assert "synthetic" in status["weather"].lower()
    assert all("synthetic lane" in s["route_source"] for s in status["lanes"].values())
    assert inst.lanes == base_instance.lanes  # unchanged lane data


def test_circuit_breaker_limits_failed_calls(no_network, monkeypatch, base_instance):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(cache.requests, "get", boom)
    build_live_instance(base_instance)
    assert len(calls) == 1  # first failure trips the breaker; the rest go straight to fallback


def test_offline_flag_never_calls_network(monkeypatch, tmp_path, base_instance):
    calls = []
    monkeypatch.setattr(cache.requests, "get", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(cache, "LIVE_DIR", tmp_path / "live")
    build_live_instance(base_instance, offline=True)
    assert calls == []


def test_planner_runs_on_fallback_instance(no_network, base_instance):
    from freshroute.checker import check_plan
    from freshroute.montecarlo import compute_risk
    from freshroute.planner import solve

    inst, _ = build_live_instance(base_instance)
    risk = compute_risk(inst, n=500)
    plan = solve(inst, risk, alpha=0.15)
    assert plan.status == "Optimal" and check_plan(inst, plan, risk).empty


def test_price_replay_and_forecast():
    hist, label = load_replay()
    assert "REPLAY (not live)" in label
    fc = forecast_next(hist, "tomato", horizon=7)
    assert len(fc) == 7 and (fc.lo <= fc.forecast).all() and (fc.forecast <= fc.hi).all()
