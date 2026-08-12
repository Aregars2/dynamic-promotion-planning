from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import dynamic_promotion_planning.policy as policy_module
from dynamic_promotion_planning.policy import (
    PlanningSpec,
    build_schedule_system,
    capacity_usage,
    evaluate_schedule_map,
    save_pickle,
    load_pickle,
    simulate_myopic_category,
    solve_dynamic_category,
)
from tests.reference_policy import brute_force_dynamic_reference


def _single_draw_problem(
    *,
    planning: PlanningSpec,
    psi: float,
    persistence: float,
    demand_factor,
    gamma: float = 0.3,
    epsilon: float = 1.5,
    price: float = 2.0,
    cost: float = 1.0,
):
    draws = {
        "A": {
            "epsilon": np.array([epsilon]),
            "gamma": np.array([gamma]),
            "psi": np.array([psi]),
            "r": np.array([persistence]),
            "weights": np.array([1.0]),
            "base_demand": np.array([10.0]),
            "regular_price": np.array([price]),
            "unit_cost": np.array([cost]),
        }
    }
    profile = {
        "A": {
            "baseline_demand": 10.0 * np.asarray(demand_factor, dtype=float),
            "demand_factor": np.asarray(demand_factor, dtype=float),
            "price_factor": np.ones(planning.evaluation_horizon),
            "cost_factor": np.ones(planning.evaluation_horizon),
            "source_week": np.arange(planning.evaluation_horizon),
        }
    }
    actions = {"A": (0.0, 0.2)}
    return draws, profile, actions


def test_all_zero_actions_produce_identical_zero_calendars() -> None:
    planning = PlanningSpec(
        decision_horizon=3,
        washout_horizon=2,
        cooldown=1,
        max_promotions=2,
        discount_factor=1.0,
    )
    draws, profiles, _ = _single_draw_problem(
        planning=planning,
        psi=0.5,
        persistence=0.5,
        demand_factor=np.ones(planning.evaluation_horizon),
    )
    actions = {"A": (0.0,)}
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[1.0]
    )
    dynamic = solve_dynamic_category(system, alpha=1.0, capacity=1)
    myopic = simulate_myopic_category(
        draws, profiles, actions, planning, alpha=1.0, capacity=1
    )
    np.testing.assert_array_equal(dynamic["schedule_map"]["A"], np.zeros(3))
    np.testing.assert_array_equal(myopic["schedule_map"]["A"], np.zeros(3))
    assert dynamic["best_value"] == pytest.approx(myopic["total_profit"], abs=1e-10)


def test_no_displacement_placebo_dynamic_equals_myopic() -> None:
    planning = PlanningSpec(
        decision_horizon=3,
        washout_horizon=0,
        cooldown=0,
        max_promotions=3,
        discount_factor=1.0,
    )
    draws, profiles, actions = _single_draw_problem(
        planning=planning,
        psi=0.0,
        persistence=0.0,
        demand_factor=np.ones(3),
    )
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[1.8]
    )
    dynamic = solve_dynamic_category(system, alpha=1.8, capacity=1)
    myopic = simulate_myopic_category(
        draws, profiles, actions, planning, alpha=1.8, capacity=1
    )
    assert dynamic["best_value"] == pytest.approx(myopic["total_profit"], abs=1e-6)
    np.testing.assert_allclose(
        dynamic["schedule_map"]["A"], myopic["schedule_map"]["A"], atol=0.0
    )


def test_strong_displacement_makes_dynamic_delay_relative_to_myopic() -> None:
    planning = PlanningSpec(
        decision_horizon=3,
        washout_horizon=2,
        cooldown=1,
        max_promotions=1,
        discount_factor=1.0,
    )
    draws, profiles, actions = _single_draw_problem(
        planning=planning,
        psi=2.5,
        persistence=0.9,
        demand_factor=np.array([1.0, 5.0, 1.0, 1.0, 1.0]),
        gamma=0.45,
    )
    alpha = 1.8
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[alpha]
    )
    dynamic = solve_dynamic_category(system, alpha=alpha, capacity=1)
    myopic = simulate_myopic_category(
        draws, profiles, actions, planning, alpha=alpha, capacity=1
    )
    assert myopic["schedule_map"]["A"][0] > 0
    assert dynamic["schedule_map"]["A"][0] == 0
    assert dynamic["best_value"] > myopic["total_profit"] + 1e-6


def test_exact_tie_allows_zero_best_second_gap() -> None:
    planning = PlanningSpec(
        decision_horizon=2,
        washout_horizon=0,
        cooldown=0,
        max_promotions=1,
        discount_factor=1.0,
    )
    draws, profiles, actions = _single_draw_problem(
        planning=planning,
        psi=0.0,
        persistence=0.0,
        demand_factor=np.ones(2),
        gamma=0.0,
        epsilon=0.0,
        price=1.0,
        cost=1.0,
    )
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[1.0]
    )
    solution = solve_dynamic_category(
        system, alpha=1.0, capacity=1, compute_second_best=True
    )
    assert solution["best_value"] == pytest.approx(0.0, abs=1e-8)
    assert solution["second_best_value"] == pytest.approx(0.0, abs=1e-8)
    assert solution["best_second_gap"] == pytest.approx(0.0, abs=1e-8)


def test_second_best_solver_failure_returns_nan(monkeypatch, tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[1.0]
    )
    real_milp = policy_module.milp
    calls = {"count": 0}

    def controlled_milp(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return real_milp(*args, **kwargs)
        return SimpleNamespace(success=False, x=None, message="forced second failure")

    monkeypatch.setattr(policy_module, "milp", controlled_milp)
    solution = solve_dynamic_category(
        system, alpha=1.0, capacity=1, compute_second_best=True
    )
    assert np.isnan(solution["second_best_value"])
    assert np.isnan(solution["best_second_gap"])


def test_primary_solver_failure_raises(monkeypatch, tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[1.0]
    )

    def failed_milp(*args, **kwargs):
        return SimpleNamespace(success=False, x=None, message="forced failure")

    monkeypatch.setattr(policy_module, "milp", failed_milp)
    with pytest.raises(RuntimeError, match="forced failure"):
        solve_dynamic_category(system, alpha=1.0, capacity=1)


def test_longer_washout_reduces_terminal_state() -> None:
    schedule = {"A": np.array([0.0, 0.0, 0.2])}
    terminal_states = []
    for washout in [0, 2, 5]:
        planning = PlanningSpec(
            decision_horizon=3,
            washout_horizon=washout,
            cooldown=1,
            max_promotions=1,
            discount_factor=1.0,
        )
        draws, profiles, _ = _single_draw_problem(
            planning=planning,
            psi=1.0,
            persistence=0.6,
            demand_factor=np.ones(planning.evaluation_horizon),
        )
        detail = evaluate_schedule_map(
            schedule, draws, profiles, planning, alpha=1.5
        )
        terminal_states.append(detail["terminal_state"]["A"])
    assert terminal_states[0] > terminal_states[1] > terminal_states[2]


def test_draw_specific_state_replay_with_heterogeneous_persistence(tiny_problem) -> None:
    planning, draws, profiles, _, _ = tiny_problem
    schedule = {
        "A": np.array([0.2, 0.0, 0.0, 0.0]),
        "B": np.zeros(planning.decision_horizon),
    }
    detail = evaluate_schedule_map(schedule, draws, profiles, planning, alpha=1.0)

    # Deliberately incorrect alternative: propagate only one weighted-average state.
    weights = draws["A"]["weights"]
    average_r = float(draws["A"]["r"] @ weights)
    average_psi = float(draws["A"]["psi"] @ weights)
    state = 0.0
    wrong_total = 0.0
    for week in range(planning.evaluation_horizon):
        depth = float(schedule["A"][week]) if week < planning.decision_horizon else 0.0
        mean_base = float(draws["A"]["base_demand"] @ weights)
        mean_epsilon = float(draws["A"]["epsilon"] @ weights)
        mean_gamma = float(draws["A"]["gamma"] @ weights)
        demand = (
            mean_base
            * profiles["A"]["demand_factor"][week]
            * max(1.0 - depth, 1e-8) ** (-mean_epsilon)
            * np.exp(mean_gamma * (depth > 0) - average_psi * state)
        )
        mean_price = float(draws["A"]["regular_price"] @ weights)
        mean_cost = float(draws["A"]["unit_cost"] @ weights)
        wrong_total += planning.discount_factor**week * (
            mean_price * (1.0 - depth) - mean_cost + 1.0 * depth
        ) * demand
        state = average_r * state + depth

    assert abs(detail["product_profit"]["A"] - wrong_total) > 1e-4


def test_pickle_round_trip(tmp_path, tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[1.0]
    )
    path = tmp_path / "schedule_system.pkl"
    save_pickle(system, path)
    loaded = load_pickle(path)
    assert loaded["input_fingerprint"] == system["input_fingerprint"]
    assert loaded["products"] == system["products"]
    for upc in loaded["products"]:
        np.testing.assert_allclose(
            loaded["product_artifacts"][upc]["schedules"],
            system["product_artifacts"][upc]["schedules"],
        )


def test_nonbinding_capacity_matches_independent_brute_force(tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    alpha = 1.3
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[alpha]
    )
    production = solve_dynamic_category(
        system, alpha=alpha, capacity=len(draws)
    )
    reference = brute_force_dynamic_reference(
        draws, profiles, actions, planning, alpha, capacity=len(draws)
    )
    assert production["best_value"] == pytest.approx(
        reference["best_value"], abs=1e-6
    )
    assert np.all(capacity_usage(production["schedule_map"]) <= len(draws))
