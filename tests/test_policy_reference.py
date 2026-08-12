from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dynamic_promotion_planning.policy import (
    analyze_policy_pair,
    build_schedule_system,
    capacity_usage,
    enumerate_feasible_schedules,
    evaluate_schedule_map,
    evaluate_schedules,
    run_policy_grid,
    schedule_input_fingerprint,
    simulate_myopic_category,
    solve_dynamic_category,
)
from tests.reference_policy import (
    brute_force_dynamic_reference,
    enumerate_schedules_reference,
    myopic_reference,
    replay_reference,
)


def _schedule_keys(schedules):
    return {
        tuple(np.round(np.asarray(schedule, dtype=float), 8))
        for schedule in schedules
    }


def test_cooldown_enumerator_matches_independent_reference() -> None:
    production = enumerate_feasible_schedules(
        actions=(0.0, 0.1, 0.2),
        decision_horizon=5,
        cooldown=2,
        max_promotions=2,
    )
    reference = enumerate_schedules_reference(
        (0.0, 0.1, 0.2),
        decision_horizon=5,
        cooldown=2,
        max_promotions=2,
    )
    assert _schedule_keys(production) == _schedule_keys(reference)
    assert (0.1, 0.0, 0.0, 0.2, 0.0) in _schedule_keys(production)
    assert (0.1, 0.0, 0.2, 0.0, 0.0) not in _schedule_keys(production)


def test_fixed_schedule_replay_matches_independent_reference(tiny_problem) -> None:
    planning, draws, profiles, _, _ = tiny_problem
    schedule_map = {
        "A": np.array([0.1, 0.0, 0.0, 0.2]),
        "B": np.array([0.0, 0.1, 0.0, 0.0]),
    }
    production = evaluate_schedule_map(
        schedule_map, draws, profiles, planning, alpha=1.7
    )
    reference = replay_reference(
        schedule_map, draws, profiles, planning, alpha=1.7
    )
    assert production["total_profit"] == pytest.approx(
        reference["total_profit"], abs=1e-10
    )
    for upc in draws:
        assert production["product_profit"][upc] == pytest.approx(
            reference["product_profit"][upc], abs=1e-10
        )
        assert production["terminal_state"][upc] == pytest.approx(
            reference["terminal_state"][upc], abs=1e-10
        )


def test_affine_schedule_values_match_direct_replay(tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    schedules = enumerate_feasible_schedules(
        actions["A"],
        planning.decision_horizon,
        planning.cooldown,
        planning.max_promotions,
    )
    values = evaluate_schedules(schedules, draws["A"], profiles["A"], planning)
    for alpha in (0.0, 0.7, 2.3):
        for row in values.iloc[:: max(1, len(values) // 7)].itertuples(index=False):
            schedule = schedules[int(row.schedule_index)]
            direct = evaluate_schedule_map(
                {"A": schedule},
                {"A": draws["A"]},
                {"A": profiles["A"]},
                planning,
                alpha,
            )["total_profit"]
            affine = float(row.intercept + alpha * row.exposure)
            assert direct == pytest.approx(affine, abs=1e-9)


@pytest.mark.parametrize("alpha", [0.0, 0.8, 1.6, 2.5])
@pytest.mark.parametrize("capacity", [1, 2])
def test_milp_and_pruning_match_independent_brute_force(
    tiny_problem,
    alpha: float,
    capacity: int,
) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    system = build_schedule_system(
        draws,
        profiles,
        actions,
        planning,
        alpha_grid=[0.0, 0.8, 1.6, 2.5],
        batch_size=64,
    )
    production = solve_dynamic_category(
        system, alpha=alpha, capacity=capacity, compute_second_best=True
    )
    reference = brute_force_dynamic_reference(
        draws, profiles, actions, planning, alpha, capacity
    )
    assert production["best_value"] == pytest.approx(
        reference["best_value"], abs=1e-6
    )
    replayed = replay_reference(
        production["schedule_map"], draws, profiles, planning, alpha
    )
    assert replayed["total_profit"] == pytest.approx(
        reference["best_value"], abs=1e-6
    )


def test_myopic_implementation_matches_independent_reference(tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    production = simulate_myopic_category(
        draws, profiles, actions, planning, alpha=1.4, capacity=1
    )
    reference = myopic_reference(
        draws, profiles, actions, planning, alpha=1.4, capacity=1
    )
    for upc in draws:
        np.testing.assert_allclose(
            production["schedule_map"][upc],
            reference["schedule_map"][upc],
            atol=0.0,
            rtol=0.0,
        )
    assert production["total_profit"] == pytest.approx(
        reference["total_profit"], abs=1e-10
    )


def test_myopic_schedule_replay_identity(tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    myopic = simulate_myopic_category(
        draws, profiles, actions, planning, alpha=1.2, capacity=1
    )
    replay = evaluate_schedule_map(
        myopic["schedule_map"], draws, profiles, planning, alpha=1.2
    )
    assert myopic["total_profit"] == pytest.approx(
        replay["total_profit"], abs=1e-10
    )
    pd.testing.assert_series_equal(
        myopic["product_profit"].sort_index(),
        replay["product_profit"].rename("myopic_profit").sort_index(),
        check_exact=False,
        atol=1e-10,
        rtol=1e-10,
    )


def test_capacity_constraints_hold_for_both_policies(tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[1.5]
    )
    dynamic = solve_dynamic_category(system, alpha=1.5, capacity=1)
    myopic = simulate_myopic_category(
        draws, profiles, actions, planning, alpha=1.5, capacity=1
    )
    assert np.all(capacity_usage(dynamic["schedule_map"]) <= 1)
    assert np.all(capacity_usage(myopic["schedule_map"]) <= 1)


def test_decompositions_reconcile_exactly(tiny_problem) -> None:
    planning, draws, profiles, actions, support = tiny_problem
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=[1.4]
    )
    dynamic = solve_dynamic_category(system, alpha=1.4, capacity=1)
    myopic = simulate_myopic_category(
        draws, profiles, actions, planning, alpha=1.4, capacity=1
    )
    summary, product, weekly = analyze_policy_pair(
        dynamic,
        myopic,
        draws,
        profiles,
        planning,
        support,
        alpha=1.4,
        capacity=1,
    )
    assert product["vdo_contribution"].sum() == pytest.approx(
        summary["vdo"], abs=1e-9
    )
    assert weekly["vdo_contribution"].sum() == pytest.approx(
        summary["vdo"], abs=1e-9
    )
    assert weekly["cumulative_vdo"].iloc[-1] == pytest.approx(
        summary["vdo"], abs=1e-9
    )


def test_dynamic_dominance_and_capacity_monotonicity_on_grid(tiny_problem) -> None:
    planning, draws, profiles, actions, support = tiny_problem
    alphas = [0.0, 0.8, 1.6, 2.5]
    capacities = [1, 2]
    system = build_schedule_system(
        draws, profiles, actions, planning, alpha_grid=alphas
    )
    run = run_policy_grid(
        system,
        draws,
        profiles,
        actions,
        support,
        alpha_values=alphas,
        capacities=capacities,
        compute_second_best=False,
    )
    results = run["results"]
    assert (results["dynamic_profit"] >= results["myopic_profit"] - 1e-6).all()
    wide = results.pivot(index="alpha", columns="capacity", values="dynamic_profit")
    assert (wide[2] >= wide[1] - 1e-6).all()


def test_fingerprint_covers_all_decision_relevant_inputs(tiny_problem) -> None:
    planning, draws, profiles, actions, _ = tiny_problem

    def fingerprint(d=draws, p=profiles, a=actions, spec=planning, grid=(1.0,)):
        return schedule_input_fingerprint(d, p, a, spec, grid)

    baseline = fingerprint()

    changed_draws = {
        key: {name: np.asarray(value).copy() for name, value in item.items()}
        for key, item in draws.items()
    }
    changed_draws["A"]["psi"][0] += 0.01
    assert fingerprint(d=changed_draws) != baseline

    changed_profiles = {
        key: {name: np.asarray(value).copy() for name, value in item.items()}
        for key, item in profiles.items()
    }
    changed_profiles["A"]["demand_factor"][0] += 0.01
    assert fingerprint(p=changed_profiles) != baseline

    changed_actions = dict(actions)
    changed_actions["A"] = (0.0, 0.1)
    assert fingerprint(a=changed_actions) != baseline

    assert fingerprint(spec=replace(planning, washout_horizon=3)) != baseline
    assert fingerprint(grid=(1.0, 1.1)) != baseline
