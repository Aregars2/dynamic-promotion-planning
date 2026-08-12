from __future__ import annotations

import numpy as np
import pandas as pd

from dynamic_promotion_planning.policy import (
    PlanningSpec,
    build_schedule_system,
    prepare_support_table,
    schedule_input_fingerprint,
    simulate_myopic_category,
    solve_dynamic_category,
    evaluate_schedule_map,
)


def _toy_problem():
    planning = PlanningSpec(
        decision_horizon=3,
        washout_horizon=2,
        cooldown=1,
        max_promotions=2,
        discount_factor=1.0,
        alpha_min=0.0,
        alpha_max=3.0,
    )
    products = ["A", "B"]
    draws = {}
    profiles = {}
    for offset, upc in enumerate(products):
        draws[upc] = {
            "epsilon": np.array([1.5]),
            "gamma": np.array([0.2]),
            "psi": np.array([0.4]),
            "r": np.array([0.5]),
            "weights": np.array([1.0]),
            "base_demand": np.array([10.0 + offset]),
            "regular_price": np.array([2.0]),
            "unit_cost": np.array([1.0]),
        }
        profiles[upc] = {
            "baseline_demand": np.full(
                planning.evaluation_horizon, 10.0 + offset
            ),
            "demand_factor": np.ones(planning.evaluation_horizon),
            "price_factor": np.ones(planning.evaluation_horizon),
            "cost_factor": np.ones(planning.evaluation_horizon),
            "source_week": np.arange(planning.evaluation_horizon),
        }
    actions = {upc: (0.0, 0.1) for upc in products}
    return planning, draws, profiles, actions


def test_planning_horizon() -> None:
    planning = PlanningSpec(decision_horizon=12, washout_horizon=8)
    assert planning.evaluation_horizon == 20


def test_full_reimbursement_schedule_value_is_affine() -> None:
    planning, draws, profiles, _ = _toy_problem()
    schedule = {"A": np.array([0.1, 0.0, 0.0]), "B": np.array([0.0, 0.1, 0.0])}
    values = [
        evaluate_schedule_map(schedule, draws, profiles, planning, share)["total_profit"]
        for share in (0.0, 0.5, 1.0)
    ]
    assert np.isclose(values[1], 0.5 * (values[0] + values[2]))


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


def test_dynamic_value_dominates_myopic_value() -> None:
    planning, draws, profiles, actions = _toy_problem()
    system = build_schedule_system(
        draws_by_product=draws,
        weekly_profiles=profiles,
        action_sets=actions,
        planning=planning,
        alpha_grid=[1.0],
        batch_size=32,
    )
    dynamic = solve_dynamic_category(
        schedule_system=system,
        alpha=1.0,
        capacity=1,
    )
    myopic = simulate_myopic_category(
        draws_by_product=draws,
        weekly_profiles=profiles,
        action_sets=actions,
        planning=planning,
        alpha=1.0,
        capacity=1,
    )
    assert dynamic["best_value"] >= myopic["total_profit"] - 1e-6


def test_dynamic_value_is_nondecreasing_in_capacity() -> None:
    planning, draws, profiles, actions = _toy_problem()
    system = build_schedule_system(
        draws_by_product=draws,
        weekly_profiles=profiles,
        action_sets=actions,
        planning=planning,
        alpha_grid=[1.0],
        batch_size=32,
    )
    values = [
        solve_dynamic_category(
            schedule_system=system,
            alpha=1.0,
            capacity=capacity,
        )["best_value"]
        for capacity in [1, 2]
    ]
    assert values[1] >= values[0] - 1e-8


def test_schedule_fingerprint_changes_with_actions() -> None:
    planning, draws, profiles, actions = _toy_problem()
    first = schedule_input_fingerprint(
        draws,
        profiles,
        actions,
        planning,
        [1.0],
    )
    changed_actions = dict(actions)
    changed_actions["A"] = (0.0, 0.1, 0.2)
    second = schedule_input_fingerprint(
        draws,
        profiles,
        changed_actions,
        planning,
        [1.0],
    )
    assert first != second
