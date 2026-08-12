"""High-level workflow helpers for the policy notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .policy import (
    PlanningSpec,
    build_schedule_system,
    load_pickle,
    save_pickle,
    schedule_input_fingerprint,
)


def demand_only_profiles(
    profiles: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
) -> dict[str, dict[str, np.ndarray]]:
    # Keep the absolute PPML baseline and fix price and cost factors.
    output: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for upc, values in profiles.items():
        baseline_demand = np.asarray(
            values["baseline_demand"],
            dtype=float,
        ).copy()

        output[str(upc)] = {
            "baseline_demand": baseline_demand,
            # Compatibility alias only; the evaluator does not use this field.
            "demand_factor": np.ones_like(baseline_demand, dtype=float),
            "price_factor": np.ones_like(
                baseline_demand,
                dtype=float,
            ),
            "cost_factor": np.ones_like(
                baseline_demand,
                dtype=float,
            ),
            "source_week": np.asarray(
                values["source_week"],
                dtype=int,
            ).copy(),
        }

    return output

def load_or_build_schedule_system(
    *,
    cache_path: Path,
    planning: PlanningSpec,
    alpha_grid: Sequence[float],
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    action_sets: Mapping[str, Sequence[float]],
    batch_size: int = 256,
    add_new_promotion_displacement: bool = True,
) -> dict:
    # Load a current cache or rebuild it from the frozen inputs.
    expected_fingerprint = (
        schedule_input_fingerprint(
            draws_by_product=draws_by_product,
            weekly_profiles=weekly_profiles,
            action_sets=action_sets,
            planning=planning,
            alpha_grid=alpha_grid,
            add_new_promotion_displacement=add_new_promotion_displacement,
        )
    )

    if cache_path.is_file():
        cached = load_pickle(cache_path)

        if (
            cached.get("input_fingerprint")
            == expected_fingerprint
        ):
            print(
                "Loaded current cache:",
                cache_path.name,
            )
            return cached

    schedule_system = build_schedule_system(
        draws_by_product=draws_by_product,
        weekly_profiles=weekly_profiles,
        action_sets=action_sets,
        planning=planning,
        alpha_grid=alpha_grid,
        batch_size=batch_size,
        add_new_promotion_displacement=add_new_promotion_displacement,
    )

    save_pickle(
        schedule_system,
        cache_path,
    )

    print(
        "Built schedule system:",
        cache_path.name,
    )

    return schedule_system

def select_washout_horizon(
    washout_results,
    *,
    vdo_stability_tolerance: float,
    terminal_state_tolerance: float,
) -> int:
    """Select the earliest horizon satisfying both pre-specified criteria.

    If no candidate satisfies both criteria, return the longest evaluated horizon.
    """
    ordered = washout_results.sort_values("washout_horizon").reset_index(drop=True)
    ordered["vdo_change"] = ordered["vdo"].diff().abs()
    eligible = ordered.loc[
        ordered["vdo_change"].le(vdo_stability_tolerance)
        & ordered["terminal_state_max"].le(terminal_state_tolerance)
    ]
    if not eligible.empty:
        return int(eligible.iloc[0]["washout_horizon"])
    return int(ordered["washout_horizon"].max())
