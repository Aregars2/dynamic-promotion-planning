from __future__ import annotations

import numpy as np

from dynamic_promotion_planning.policy import PlanningSpec, demand_multiplier_audit


def test_multiplier_audit_accepts_scalar_product_week_baseline():
    planning = PlanningSpec(decision_horizon=1, washout_horizon=1)
    draws = {"A": {
        "epsilon": np.array([1.0, 2.0]), "gamma": np.array([0.2, 0.3]),
        "psi": np.array([0.1, 0.2]), "r": np.array([0.5, 0.7]),
        "weights": np.array([0.4, 0.6]), "base_demand": np.array([5.0, 8.0]),
        "regular_price": np.array([2.0, 2.0]), "unit_cost": np.array([1.0, 1.0]),
    }}
    profiles = {"A": {"baseline_demand": np.array([10.0, 11.0]), "price_factor": np.ones(2), "cost_factor": np.ones(2)}}
    audit = demand_multiplier_audit({"A": np.array([0.1])}, draws, profiles, planning)
    assert audit["q_base"].tolist() == [10.0, 11.0]
    assert np.isfinite(audit["predicted_demand"]).all()
