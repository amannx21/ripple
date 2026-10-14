"""Plan container shared by the optimizer and the baselines."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

PLAN_COLUMNS = [
    "batch_id", "commodity", "origin", "action", "dest_id", "lane_id", "discount", "quantity_kg",
    "priority_rank", "sar", "p_ok", "exp_net_inr", "reason",
]
ACTIONS = ("SHIP", "DISCOUNT", "HOLD", "DIVERT")


@dataclass
class Plan:
    method: str
    rows: pd.DataFrame
    trucks: dict[str, int]
    alpha: float
    objective: float = float("nan")
    solve_time_s: float = 0.0
    status: str = "ok"
    meta: dict = field(default_factory=dict)

    @property
    def shipments(self) -> pd.DataFrame:
        return self.rows[self.rows.action.isin(["SHIP", "DISCOUNT"])]

    def to_json(self) -> str:
        return json.dumps({
            "method": self.method, "alpha": self.alpha, "objective_expected_net_inr": self.objective,
            "solve_time_s": self.solve_time_s, "status": self.status, "trucks": self.trucks, "meta": self.meta,
            "rows": self.rows.to_dict(orient="records"),
        }, indent=2, default=float)


def empty_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=PLAN_COLUMNS)
