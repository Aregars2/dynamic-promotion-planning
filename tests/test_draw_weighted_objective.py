from __future__ import annotations

import numpy as np
import pandas as pd

from dynamic_promotion_planning.calibration import CalibrationConfig, _expand_persistence_draws
from dynamic_promotion_planning.policy import (
    PlanningSpec,
    audit_draw_weighted_schedule_values,
    build_schedule_system,
)


def _draw_problem():
    planning = PlanningSpec(decision_horizon=2, washout_horizon=0, cooldown=0, max_promotions=1)
    draws = {"A": {
        "epsilon": np.array([1.0, 2.0]), "gamma": np.array([0.1, 0.5]),
        "psi": np.array([0.2, 0.8]), "r": np.array([0.3, 0.7]),
        "weights": np.array([0.25, 0.75]), "base_demand": np.array([10.0, 20.0]),
        "regular_price": np.array([2.0, 2.0]), "unit_cost": np.array([1.0, 1.0]),
    }}
    profiles = {"A": {"baseline_demand": np.array([15.0, 15.0]), "price_factor": np.ones(2), "cost_factor": np.ones(2)}}
    return planning, draws, profiles, {"A": (0.0, 0.2)}


def test_pi_n_and_pi_d_optimize_weighted_draw_values_and_share_weights():
    planning, draws, profiles, actions = _draw_problem()
    dynamic = build_schedule_system(draws, profiles, actions, planning, [1.0])
    naive = build_schedule_system(draws, profiles, actions, planning, [1.0], add_new_promotion_displacement=False)
    assert np.isclose(draws["A"]["weights"].sum(), 1.0)
    for system in (dynamic, naive):
        audit = audit_draw_weighted_schedule_values(system, draws, profiles, upc="A", alpha=1.0, capacity=1)
        by_schedule = audit.drop_duplicates("schedule_index")
        np.testing.assert_allclose(by_schedule["weighted_expected_schedule_value"], by_schedule["stored_schedule_value"])
        selected = by_schedule.loc[by_schedule.selected_by_optimizer, "stored_schedule_value"].iloc[0]
        assert selected == by_schedule["stored_schedule_value"].max()
    np.testing.assert_array_equal(draws["A"]["weights"], draws["A"]["weights"])


def test_persistence_expansion_divides_each_bootstrap_draw_weight():
    base = pd.DataFrame({
        "upc": ["A", "A"], "bootstrap_draw": [0, 1],
        "inventory_persistence": [0.4, 0.5], "post1_depth_slope": [np.nan, np.nan],
        "post2_depth_slope": [np.nan, np.nan],
    })
    config = CalibrationConfig(bootstrap_replications=2, persistence_grid=(0.2, 0.6))
    expanded = _expand_persistence_draws(base, config)
    assert np.isclose(expanded.draw_weight.sum(), 1.0)
    counts = expanded.groupby("bootstrap_draw", observed=True).size()
    totals = expanded.groupby("bootstrap_draw", observed=True).draw_weight.sum()
    np.testing.assert_allclose(totals.to_numpy(), np.full(len(totals), 0.5))
    np.testing.assert_allclose(expanded.draw_weight.to_numpy(), 1.0 / (2 * counts.reindex(expanded.bootstrap_draw).to_numpy()))
