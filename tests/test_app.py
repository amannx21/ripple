"""Dashboard smoke test: every page renders offline without exceptions or error boxes."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

PAGES = ["Home.py"] + sorted(p.name for p in (APP / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from freshroute.live import cache

    monkeypatch.setattr(cache, "_down_until", [1e18])  # never touch the network in tests
    path = APP / page if page == "Home.py" else APP / "pages" / page
    at = AppTest.from_file(str(path), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_replan_button_heatwave(monkeypatch):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "pages" / "4_Simulator_and_Replan.py"), default_timeout=240)
    at.run()
    heat = [b for b in at.button if "Heatwave" in b.label][0]
    heat.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("What changed" in s.value for s in at.subheader)
