from __future__ import annotations

import numpy as np
import pandas as pd

from dynamic_promotion_planning.policy import (
    PlanningSpec,
    _weekly_demand_summary,
    evaluate_schedule_map,
    schedule_input_fingerprint,
)


def test_baseline_revenue_is_literal_store_sum() -> None:
    predictions = pd.DataFrame(
        {
            "upc": ["A", "A", "A"],
            "week": [1, 1, 2],
            "regular_price": [2.0, 3.0, 2.5],
            "mu_hat_regular_counterfactual": [10.0, 5.0, 8.0],
        }
    )
    summary = _weekly_demand_summary(predictions).set_index(["upc", "week"])
    assert summary.loc[("A", 1), "counterfactual_baseline_demand"] == 15.0
    assert summary.loc[("A", 1), "counterfactual_baseline_revenue"] == 35.0
    assert summary.loc[("A", 2), "counterfactual_baseline_revenue"] == 20.0


def test_aggregated_revenue_profit_equals_explicit_store_sum() -> None:
    planning = PlanningSpec(decision_horizon=1, washout_horizon=0)
    draws = {
        "A": {
            "epsilon": np.array([1.2]), "gamma": np.array([0.3]),
            "psi": np.array([0.2]), "r": np.array([0.5]), "weights": np.array([1.0]),
            "base_demand": np.array([1.0]), "regular_price": np.array([2.0]),
            "unit_cost": np.array([1.1]),
        }
    }
    profile = {
        "A": {
            "baseline_demand": np.array([15.0]),
            "baseline_revenue": np.array([35.0]),
            "price_factor": np.ones(1), "cost_factor": np.ones(1),
        }
    }
    depth, reimbursement = 0.2, 0.4
    value = evaluate_schedule_map({"A": np.array([depth])}, draws, profile, planning, reimbursement)["total_profit"]
    multiplier = (1.0 - depth) ** -1.2 * np.exp(0.3)
    explicit = multiplier * sum(
        ((1.0 - depth + reimbursement * depth) * price - 1.1) * quantity
        for price, quantity in [(2.0, 10.0), (3.0, 5.0)]
    )
    np.testing.assert_allclose(value, explicit)


def test_schedule_fingerprint_changes_with_revenue_baseline() -> None:
    planning = PlanningSpec(decision_horizon=1, washout_horizon=0)
    draws = {"A": {"epsilon": np.ones(1), "gamma": np.zeros(1), "psi": np.zeros(1),
                   "r": np.zeros(1), "weights": np.ones(1), "base_demand": np.ones(1),
                   "regular_price": np.full(1, 2.0), "unit_cost": np.ones(1)}}
    profile = {"A": {"baseline_demand": np.array([10.0]), "baseline_revenue": np.array([20.0]),
                     "price_factor": np.ones(1), "cost_factor": np.ones(1)}}
    first = schedule_input_fingerprint(draws, profile, {"A": (0.0, 0.1)}, planning, [0.0, 1.0])
    profile["A"]["baseline_revenue"] = np.array([25.0])
    second = schedule_input_fingerprint(draws, profile, {"A": (0.0, 0.1)}, planning, [0.0, 1.0])
    assert first != second
