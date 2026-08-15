from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_promotion_planning.policy import PlanningSpec


@pytest.fixture
def tiny_problem():
    planning = PlanningSpec(
        decision_horizon=4,
        washout_horizon=2,
        cooldown=1,
        max_promotions=2,
        discount_factor=0.97,
        alpha_min=0.0,
        alpha_max=3.0,
    )
    draws = {
        "A": {
            "epsilon": np.array([1.2, 1.8]),
            "gamma": np.array([0.15, 0.30]),
            "psi": np.array([0.25, 0.60]),
            "r": np.array([0.30, 0.70]),
            "weights": np.array([0.4, 0.6]),
            "base_demand": np.array([10.0, 12.0]),
            "regular_price": np.array([2.2, 2.2]),
            "unit_cost": np.array([1.0, 1.0]),
        },
        "B": {
            "epsilon": np.array([1.0]),
            "gamma": np.array([0.20]),
            "psi": np.array([0.40]),
            "r": np.array([0.55]),
            "weights": np.array([1.0]),
            "base_demand": np.array([9.0]),
            "regular_price": np.array([2.0]),
            "unit_cost": np.array([0.9]),
        },
    }
    profiles = {
        "A": {
            "baseline_demand": np.array([10.0, 9.0, 14.0, 11.0, 10.0, 10.0]),
            "baseline_revenue": np.array([22.0, 19.8, 30.8, 24.2, 22.0, 22.0]),
            "demand_factor": np.array([1.0, 0.9, 1.4, 1.1, 1.0, 1.0]),
            "price_factor": np.ones(6),
            "cost_factor": np.ones(6),
            "source_week": np.arange(100, 106),
        },
        "B": {
            "baseline_demand": np.array([9.9, 11.7, 7.2, 9.0, 9.0, 9.0]),
            "baseline_revenue": np.array([19.8, 23.4, 14.4, 18.0, 18.0, 18.0]),
            "demand_factor": np.array([1.1, 1.3, 0.8, 1.0, 1.0, 1.0]),
            "price_factor": np.ones(6),
            "cost_factor": np.ones(6),
            "source_week": np.arange(100, 106),
        },
    }
    actions = {"A": (0.0, 0.10, 0.20), "B": (0.0, 0.10)}
    support = pd.DataFrame(
        {
            "upc": ["A", "A", "B"],
            "depth_cluster": [0.10, 0.20, 0.10],
            "observations": [30, 20, 25],
            "panels": [5, 4, 5],
        }
    )
    return planning, draws, profiles, actions, support
