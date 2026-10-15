"""End-to-end: the reproduce pipeline runs (quick mode) and is deterministic."""
import json

from freshroute.config import RESULTS_DIR


def test_quick_reproduce_runs_and_is_deterministic():
    from freshroute.reproduce import main

    a = main(["--quick", "--workers", "3", "--data", "synthetic"])
    first = (RESULTS_DIR / "quick" / "scorecard.json").read_text()
    b = main(["--quick", "--workers", "2", "--data", "synthetic"])
    second = (RESULTS_DIR / "quick" / "scorecard.json").read_text()
    assert first == second, "two runs with the same seeds must give identical scorecards"
    assert a["quick_mode"] and a["heldout"]["summary"]["FreshRoute planner"]["violations_mean"] == 0
    seeds = a["seed_sets"]
    assert seeds["tuning"][1] < seeds["heldout"][0], "tuning and held-out seed ranges must not overlap"
    assert json.loads(first)["data_source"] == "SYNTHETIC"
