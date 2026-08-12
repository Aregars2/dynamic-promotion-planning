"""Independent tiny-problem reference implementation for policy tests.

This module intentionally does not import ``dynamic_promotion_planning.policy``.
It provides direct week-by-week replay, recursive schedule enumeration,
exhaustive category optimization, and a direct myopic rule.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np


def enumerate_schedules_reference(
    actions: Sequence[float],
    *,
    decision_horizon: int,
    cooldown: int,
    max_promotions: int,
) -> list[np.ndarray]:
    """Enumerate feasible schedules recursively.

    ``cooldown=2`` means that two zero-action weeks must follow a promotion.
    """
    action_values = tuple(sorted({float(value) for value in actions}))
    if 0.0 not in action_values:
        action_values = (0.0, *action_values)

    schedules: list[np.ndarray] = []

    def recurse(
        week: int,
        schedule: np.ndarray,
        last_promotion: int,
        promotion_count: int,
    ) -> None:
        if week == decision_horizon:
            schedules.append(schedule.copy())
            return

        for depth in action_values:
            if depth > 0:
                if promotion_count >= max_promotions:
                    continue
                if week - last_promotion <= cooldown:
                    continue
            schedule[week] = depth
            recurse(
                week + 1,
                schedule,
                week if depth > 0 else last_promotion,
                promotion_count + int(depth > 0),
            )
        schedule[week] = 0.0

    recurse(
        week=0,
        schedule=np.zeros(decision_horizon, dtype=float),
        last_promotion=-10_000,
        promotion_count=0,
    )
    return schedules


def replay_reference(
    schedule_map: Mapping[str, Sequence[float]],
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    planning: Any,
    alpha: float,
) -> dict[str, Any]:
    """Evaluate a fixed schedule with explicit draw-specific state propagation."""
    product_profit: dict[str, float] = {}
    terminal_state: dict[str, float] = {}
    weekly_rows: list[dict[str, float | int | str | bool]] = []

    for upc in sorted(draws_by_product):
        draws = draws_by_product[upc]
        profile = weekly_profiles[upc]
        state = np.zeros(len(draws["weights"]), dtype=float)
        schedule = np.asarray(schedule_map[upc], dtype=float)
        total = 0.0

        for week in range(planning.evaluation_horizon):
            depth = float(schedule[week]) if week < planning.decision_horizon else 0.0
            demand = (
                np.asarray(profile.get("baseline_demand", draws["base_demand"]), dtype=float)[week]
                * max(1.0 - depth, 1e-8)
                ** (-np.asarray(draws["epsilon"], dtype=float))
                * np.exp(
                    np.asarray(draws["gamma"], dtype=float) * (depth > 0)
                    - np.asarray(draws["psi"], dtype=float) * state
                )
            )
            price = (
                np.asarray(draws["regular_price"], dtype=float)
                * float(profile["price_factor"][week])
            )
            cost = (
                np.asarray(draws["unit_cost"], dtype=float)
                * float(profile["cost_factor"][week])
            )
            reimbursement = alpha * np.asarray(draws["regular_price"], dtype=float) * depth
            profit_draw = (price * (1.0 - depth) - cost + reimbursement) * demand
            expected_profit = float(
                profit_draw @ np.asarray(draws["weights"], dtype=float)
            )
            discounted_profit = planning.discount_factor**week * expected_profit
            total += discounted_profit
            weekly_rows.append(
                {
                    "upc": upc,
                    "week": week + 1,
                    "decision_week": week < planning.decision_horizon,
                    "action": depth,
                    "profit": discounted_profit,
                    "expected_demand": float(
                        demand @ np.asarray(draws["weights"], dtype=float)
                    ),
                    "inventory_state_before": float(
                        state @ np.asarray(draws["weights"], dtype=float)
                    ),
                }
            )
            state = np.asarray(draws["r"], dtype=float) * state + depth

        product_profit[upc] = total
        terminal_state[upc] = float(
            state @ np.asarray(draws["weights"], dtype=float)
        )

    return {
        "product_profit": product_profit,
        "terminal_state": terminal_state,
        "weekly_rows": weekly_rows,
        "total_profit": float(sum(product_profit.values())),
    }


def brute_force_dynamic_reference(
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    action_sets: Mapping[str, Sequence[float]],
    planning: Any,
    alpha: float,
    capacity: int,
) -> dict[str, Any]:
    """Exhaustively solve the tiny category problem."""
    products = sorted(draws_by_product)
    product_schedules = {
        upc: enumerate_schedules_reference(
            action_sets[upc],
            decision_horizon=planning.decision_horizon,
            cooldown=planning.cooldown,
            max_promotions=planning.max_promotions,
        )
        for upc in products
    }

    feasible_values: list[tuple[float, tuple[np.ndarray, ...]]] = []
    for combination in product(*(product_schedules[upc] for upc in products)):
        occupancy = np.sum(
            np.vstack([np.asarray(schedule) > 0 for schedule in combination]),
            axis=0,
        )
        if np.any(occupancy > capacity):
            continue
        schedule_map = {
            upc: np.asarray(schedule, dtype=float)
            for upc, schedule in zip(products, combination)
        }
        value = replay_reference(
            schedule_map,
            draws_by_product,
            weekly_profiles,
            planning,
            alpha,
        )["total_profit"]
        feasible_values.append((float(value), combination))

    if not feasible_values:
        raise RuntimeError("Reference problem has no feasible category schedule.")

    feasible_values.sort(key=lambda item: item[0], reverse=True)
    best_value, best_combination = feasible_values[0]
    schedule_map = {
        upc: np.asarray(schedule, dtype=float)
        for upc, schedule in zip(products, best_combination)
    }
    second_best_value = (
        float(feasible_values[1][0]) if len(feasible_values) > 1 else np.nan
    )
    return {
        "best_value": float(best_value),
        "second_best_value": second_best_value,
        "schedule_map": schedule_map,
        "all_values": np.asarray([value for value, _ in feasible_values], dtype=float),
    }


def current_profit_reference(
    depth: float,
    state: np.ndarray,
    draws: Mapping[str, np.ndarray],
    profile: Mapping[str, np.ndarray],
    week: int,
    alpha: float,
) -> float:
    """Expected undiscounted profit in one week."""
    demand = (
        np.asarray(profile.get("baseline_demand", draws["base_demand"]), dtype=float)[week]
        * max(1.0 - depth, 1e-8)
        ** (-np.asarray(draws["epsilon"], dtype=float))
        * np.exp(
            np.asarray(draws["gamma"], dtype=float) * (depth > 0)
            - np.asarray(draws["psi"], dtype=float) * state
        )
    )
    price = (
        np.asarray(draws["regular_price"], dtype=float)
        * float(profile["price_factor"][week])
    )
    cost = (
        np.asarray(draws["unit_cost"], dtype=float)
        * float(profile["cost_factor"][week])
    )
    reimbursement = alpha * np.asarray(draws["regular_price"], dtype=float) * depth
    profit_draw = (price * (1.0 - depth) - cost + reimbursement) * demand
    return float(profit_draw @ np.asarray(draws["weights"], dtype=float))


def myopic_reference(
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    action_sets: Mapping[str, Sequence[float]],
    planning: Any,
    alpha: float,
    capacity: int,
) -> dict[str, Any]:
    """Directly simulate the documented myopic rule."""
    products = sorted(draws_by_product)
    states = {
        upc: np.zeros(len(draws_by_product[upc]["weights"]), dtype=float)
        for upc in products
    }
    last_promotion = {upc: -10_000 for upc in products}
    counts = {upc: 0 for upc in products}
    schedules = {
        upc: np.zeros(planning.decision_horizon, dtype=float)
        for upc in products
    }

    for week in range(planning.decision_horizon):
        candidates: list[tuple[float, str, float]] = []
        for upc in products:
            draws = draws_by_product[upc]
            profile = weekly_profiles[upc]
            base = current_profit_reference(
                0.0, states[upc], draws, profile, week, alpha
            )
            feasible = (
                counts[upc] < planning.max_promotions
                and week - last_promotion[upc] > planning.cooldown
            )
            if not feasible:
                continue

            positive = [
                float(value) for value in action_sets[upc] if float(value) > 0
            ]
            if not positive:
                continue
            profits = [
                current_profit_reference(
                    depth, states[upc], draws, profile, week, alpha
                )
                for depth in positive
            ]
            position = int(np.argmax(profits))
            increment = profits[position] - base
            if increment > 1e-12:
                candidates.append((float(increment), upc, positive[position]))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen = {upc: 0.0 for upc in products}
        for _, upc, depth in candidates[: int(capacity)]:
            chosen[upc] = depth
            schedules[upc][week] = depth
            last_promotion[upc] = week
            counts[upc] += 1

        for upc in products:
            states[upc] = (
                np.asarray(draws_by_product[upc]["r"], dtype=float) * states[upc]
                + chosen[upc]
            )

    detail = replay_reference(
        schedules,
        draws_by_product,
        weekly_profiles,
        planning,
        alpha,
    )
    return {
        "schedule_map": schedules,
        "total_profit": detail["total_profit"],
        "product_profit": detail["product_profit"],
        "terminal_state": detail["terminal_state"],
    }
