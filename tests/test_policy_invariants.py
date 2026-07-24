from __future__ import annotations

import pandas as pd

from price_of_extrapolation.policy import PlanningSpec, prepare_support_table


def test_planning_horizon() -> None:
    planning = PlanningSpec(decision_horizon=12, washout_horizon=8)
    assert planning.evaluation_horizon == 20


def test_support_aliases() -> None:
    raw = pd.DataFrame(
        {
            "upc": [123],
            "bin_center": [0.10],
            "support_count": [20],
            "support_panels": [4],
        }
    )
    result = prepare_support_table(raw)
    assert result.loc[0, "upc"] == "123"
    assert result.loc[0, "depth_cluster"] == 0.10
    assert result.loc[0, "observations"] == 20
    assert result.loc[0, "panels"] == 4
