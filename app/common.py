"""Shared helpers for the Streamlit pages: loading, caching, badges and the sidebar."""
from __future__ import annotations

import copy
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freshroute.adapters import AdapterError, load_base_instance, organizer  # noqa: E402
from freshroute.checker import check_plan  # noqa: E402
from freshroute.config import RESULTS_DIR  # noqa: E402
from freshroute.montecarlo import compute_risk  # noqa: E402
from freshroute.planner import solve  # noqa: E402

BADGE_CSS = """
<style>
.fr-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;margin:0 6px 4px 0;
  border:1px solid rgba(0,0,0,.08)}
.fr-syn{background:#fde7c8;color:#7a4500}
.fr-org{background:#d9ecff;color:#0b4a8b}
.fr-live{background:#d6f5e3;color:#0d6b3a}
.fr-cached{background:#e6eefc;color:#28457a}
.fr-fallback{background:#fbe0de;color:#8a1f17}
.fr-ok{background:#d6f5e3;color:#0d6b3a}
.fr-bad{background:#fbe0de;color:#8a1f17}
.fr-neutral{background:#eceff3;color:#333}
</style>
"""


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="fr-badge fr-{kind}">{text}</span>'


def source_kind(label: str) -> str:
    u = (label or "").upper()
    if u.startswith("LIVE"):
        return "live"
    if u.startswith("CACHED"):
        return "cached"
    if u.startswith("FALLBACK"):
        return "fallback"
    if "SYNTHETIC" in u:
        return "syn"
    if "ORGANIZER" in u:
        return "org"
    return "neutral"


def page_setup(title: str):
    st.set_page_config(page_title=f"FreshRoute · {title}", page_icon="🥭", layout="wide")
    st.markdown(BADGE_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading dataset…")
def base_instance(prefer: str = "auto"):
    return load_base_instance(prefer)


def data_badges(inst) -> str:
    out = []
    if inst.source_label == "SYNTHETIC":
        out.append(badge("SYNTHETIC DATA", "syn"))
    else:
        out.append(badge("ORGANIZER DATA", "org"))
    out.append(badge(f"temperatures: {inst.temp_source}", source_kind(inst.temp_source)))
    return " ".join(out)


def sidebar() -> dict:
    """Common controls; returns the chosen settings and stores them in session state."""
    s = st.session_state
    with st.sidebar:
        st.markdown("### FreshRoute controls")
        has_org = organizer.organizer_data_present()
        mode_opts = ["Offline (scored dataset)", "Live geography (OSRM + Open-Meteo)"]
        s.setdefault("mode", mode_opts[0])
        mode = st.radio("Mode", mode_opts, key="mode",
                        help="Offline uses the scored dataset with the synthetic diurnal temperature model. "
                             "Live re-routes the same lanes on real roads and uses forecast temperatures; "
                             "it falls back to bundled cached responses if the network is unavailable.")
        base = base_instance()
        default_alpha = float(_tuned_alpha() or base.settings.get("risk_alpha", 0.10))
        s.setdefault("alpha", default_alpha)
        alpha = st.slider("Risk limit α (max P(not sellable on arrival))", 0.01, 0.40, key="alpha", step=0.01,
                          help="A destination is feasible for a batch only if P(arrives sellable and on time) ≥ 1−α. "
                               f"Tuned on the tuning seeds: {default_alpha}.")
        s.setdefault("departure_hour", int(base.settings["planning"]["departure_hour"]))
        dep = st.slider("Departure hour (local)", 0, 23, key="departure_hour")
        st.caption(("Organizer dataset detected." if has_org else "No organizer files in data/organizer/: using "
                    "the SYNTHETIC dataset.") + " Prices in live mode are REPLAY (not live).")
    return {"mode": mode, "alpha": float(alpha), "departure_hour": int(dep), "live": mode.startswith("Live")}


def _tuned_alpha():
    sc = load_scorecard()
    return sc.get("risk_alpha_tuned") if sc else None


@st.cache_data(show_spinner=False)
def load_scorecard() -> dict | None:
    p = RESULTS_DIR / "scorecard.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def load_timing() -> dict:
    p = RESULTS_DIR / "timing.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_records(name: str) -> pd.DataFrame | None:
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner="Fetching routes and weather (cached, rate-limited)…", ttl=3600)
def live_instance(departure_hour: int):
    from freshroute.live import build_live_instance

    return build_live_instance(base_instance(), departure_hour=departure_hour)


def offline_instance(departure_hour: int):
    base = base_instance()
    if departure_hour == int(base.settings["planning"]["departure_hour"]):
        return base
    return _offline_with_departure(departure_hour)


@st.cache_resource(show_spinner=False)
def _offline_with_departure(departure_hour: int):
    from freshroute.temperature import build_forecasts

    inst = copy.deepcopy(base_instance())
    inst.settings = copy.deepcopy(inst.settings)
    inst.settings["planning"]["departure_hour"] = departure_hour
    inst.temp_forecast = build_forecasts(inst, departure_hour)
    return inst


def current_instance(ctl: dict):
    """(instance, live_status or None) for the sidebar choice; never raises."""
    if ctl["live"]:
        try:
            inst, status = live_instance(ctl["departure_hour"])
            return inst, status
        except Exception as exc:  # noqa: BLE001 - UI must not crash
            st.warning(f"Live mode unavailable, showing offline data instead: {exc}")
    return offline_instance(ctl["departure_hour"]), None


@st.cache_resource(show_spinner="Monte Carlo + optimizer…", max_entries=32)
def _plan_for(key: str, _inst, alpha: float):
    risk = compute_risk(_inst, keep_samples=True)
    plan = solve(_inst, risk, alpha=alpha)
    viol = check_plan(_inst, plan, risk)
    return risk, plan, viol


def plan_for(inst, alpha: float, tag: str):
    return _plan_for(f"{tag}|{alpha:.3f}", inst, alpha)


def instance_tag(ctl: dict) -> str:
    return f"{'live' if ctl['live'] else 'offline'}|dep{ctl['departure_hour']}"


def safe(fn):
    """Run a page body and show a readable error instead of a stack trace."""
    try:
        fn()
    except AdapterError as exc:
        st.error(f"Data problem: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong: {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


def money(x: float) -> str:
    if x is None or x != x:
        return "–"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}₹{x / 1e7:,.2f} Cr"
    if x >= 1e5:
        return f"{sign}₹{x / 1e5:,.2f} L"
    return f"{sign}₹{x:,.0f}"
