from __future__ import annotations

import numpy as np

from dynamic_promotion_planning.policy import (
    PlanningSpec,
    _evaluate_schedule_batch,
    build_schedule_system,
    displacement_naive_draws,
    evaluate_schedule_map,
    evaluate_schedules,
    simulate_myopic_category,
    solve_dynamic_category,
)


def _problem():
    planning = PlanningSpec(decision_horizon=2, washout_horizon=0, cooldown=0, max_promotions=2)
    draws = {"A": {
        "epsilon": np.array([1.2]), "gamma": np.array([0.3]), "psi": np.array([0.7]),
        "r": np.array([0.6]), "weights": np.array([1.0]), "base_demand": np.array([10.0]),
        "regular_price": np.array([2.0]), "unit_cost": np.array([1.0]),
    }}
    profiles = {"A": {"baseline_demand": np.full(2, 10.0), "price_factor": np.ones(2), "cost_factor": np.ones(2), "source_week": np.arange(2)}}
    return planning, draws, profiles, {"A": (0.0, 0.2)}


def test_fresh_start_exact_naive_matches_legacy_psi_zero_objective_and_calendar():
    planning, draws, profiles, actions = _problem()
    exact = build_schedule_system(draws, profiles, actions, planning, [1.0], add_new_promotion_displacement=False)
    legacy = build_schedule_system(displacement_naive_draws(draws), profiles, actions, planning, [1.0])
    exact_solution = solve_dynamic_category(exact, 1.0, 1)
    legacy_solution = solve_dynamic_category(legacy, 1.0, 1)
    assert exact_solution["best_value"] == legacy_solution["best_value"]
    np.testing.assert_array_equal(exact_solution["schedule_map"]["A"], legacy_solution["schedule_map"]["A"])


def test_candidate_promotion_updates_only_dynamic_optimization_state_and_replay_is_full():
    planning, draws, profiles, _ = _problem()
    schedules = np.array([[0.2, 0.0]])
    _, _, dynamic_terminal = _evaluate_schedule_batch(schedules, draws["A"], profiles["A"], planning)
    _, _, naive_terminal = _evaluate_schedule_batch(
        schedules, draws["A"], profiles["A"], planning, add_new_promotion_displacement=False
    )
    assert dynamic_terminal[0] == 0.2 * 0.6
    assert naive_terminal[0] == 0.0
    replay = evaluate_schedule_map({"A": schedules[0]}, draws, profiles, planning, 1.0)
    assert replay["terminal_state"]["A"] == 0.2 * 0.6


def test_all_three_policies_share_fresh_start_initialization():
    planning, draws, profiles, actions = _problem()
    dynamic = build_schedule_system(draws, profiles, actions, planning, [1.0])
    naive = build_schedule_system(draws, profiles, actions, planning, [1.0], add_new_promotion_displacement=False)
    assert dynamic["add_new_promotion_displacement"] is True
    assert naive["add_new_promotion_displacement"] is False
    myopic = simulate_myopic_category(draws, profiles, actions, planning, 1.0, 1)
    assert myopic["weekly_profit"].iloc[0]["week"] == 1
