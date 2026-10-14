"""Static figures written by the reproduce script (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {
    "FreshRoute planner": "#1f7a5a",
    "Deterministic optimizer (ablation)": "#7a9cc6",
    "Baseline A: FIFO": "#c0603a",
    "Baseline B: greedy-by-price": "#d9a441",
    "Stale plan (no replan)": "#8c8c8c",
}
SHORT = {
    "FreshRoute planner": "FreshRoute",
    "Deterministic optimizer (ablation)": "Deterministic\noptimizer",
    "Baseline A: FIFO": "FIFO\n(baseline A)",
    "Baseline B: greedy-by-price": "Greedy\n(baseline B)",
    "Stale plan (no replan)": "Stale plan",
}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def headline(df: pd.DataFrame, path: Path, label: str):
    metrics = [("spoilage_pct", "Spoilage (% of stock)"), ("fulfillment_pct", "Demand fulfilled (%)"),
               ("net_revenue", "Net revenue (INR lakh)")]
    methods = [m for m in COLORS if m in set(df.method)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (k, title) in zip(axes, metrics):
        data = [df[df.method == m][k].to_numpy() / (1e5 if k == "net_revenue" else 1) for m in methods]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, whis=(5, 95))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(methods) + 1), [SHORT[m] for m in methods], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(label, fontsize=10)
    _save(fig, path)


def tuning_curve(tune: pd.DataFrame, chosen: float, path: Path):
    t = tune[tune.alpha != "FIFO"].copy()
    t["alpha"] = t.alpha.astype(float)
    g = t.groupby("alpha")[["net_revenue", "spoilage_pct"]].mean()
    fifo = tune[tune.alpha == "FIFO"][["net_revenue", "spoilage_pct"]].mean()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(g.index, g.net_revenue / 1e5, "o-", color=COLORS["FreshRoute planner"], label="FreshRoute")
    ax[0].axhline(fifo.net_revenue / 1e5, color=COLORS["Baseline A: FIFO"], ls="--", label="FIFO")
    ax[0].axvline(chosen, color="k", alpha=0.4, ls=":")
    ax[0].set_xlabel("risk limit alpha")
    ax[0].set_ylabel("mean net revenue (INR lakh)")
    ax[0].legend()
    ax[1].plot(g.index, g.spoilage_pct, "o-", color=COLORS["FreshRoute planner"])
    ax[1].axhline(fifo.spoilage_pct, color=COLORS["Baseline A: FIFO"], ls="--")
    ax[1].axvline(chosen, color="k", alpha=0.4, ls=":")
    ax[1].set_xlabel("risk limit alpha")
    ax[1].set_ylabel("mean spoilage (%)")
    fig.suptitle(f"Tuning set only: chosen alpha = {chosen}", fontsize=10)
    _save(fig, path)


def sensitivity(sens: pd.DataFrame, path: Path):
    factors = list(dict.fromkeys(sens.factor))
    fig, axes = plt.subplots(2, len(factors), figsize=(3.2 * len(factors), 6))
    for j, f in enumerate(factors):
        s = sens[sens.factor == f]
        for m in ["FreshRoute planner", "Baseline A: FIFO"]:
            g = s[s.method == m].groupby("level")[["net_revenue", "spoilage_pct"]].mean()
            axes[0, j].plot(g.index, g.net_revenue / 1e5, "o-", color=COLORS[m], label=SHORT[m].replace("\n", " "))
            axes[1, j].plot(g.index, g.spoilage_pct, "o-", color=COLORS[m])
        axes[0, j].set_title(f.replace("_", " "), fontsize=9)
        axes[1, j].set_xlabel("level")
        axes[0, 0].set_ylabel("net revenue (INR lakh)")
        axes[1, 0].set_ylabel("spoilage (%)")
        for a in axes[:, j]:
            a.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Sensitivity (seeds 3000-3059): planner keeps default assumptions, the world's noise changes", fontsize=10)
    _save(fig, path)


def disruptions(dis: pd.DataFrame, path: Path):
    kinds = list(dict.fromkeys(dis.disruption))
    methods = [m for m in ["FreshRoute planner", "Stale plan (no replan)", "Baseline A: FIFO"] if m in set(dis.method)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    w = 0.8 / len(methods)
    x = np.arange(len(kinds))
    for i, m in enumerate(methods):
        g = dis[dis.method == m].groupby("disruption")[["net_revenue", "spoilage_pct"]].mean().reindex(kinds)
        axes[0].bar(x + i * w, g.net_revenue / 1e5, w, color=COLORS[m], label=SHORT[m].replace("\n", " "))
        axes[1].bar(x + i * w, g.spoilage_pct, w, color=COLORS[m])
    for a, t in zip(axes, ["mean net revenue (INR lakh)", "mean spoilage (%)"]):
        a.set_xticks(x + w * (len(methods) - 1) / 2, [k.replace("_", "\n") for k in kinds], fontsize=8)
        a.set_ylabel(t)
        a.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Disruption suite: replanning vs. a stale plan vs. FIFO", fontsize=10)
    _save(fig, path)
