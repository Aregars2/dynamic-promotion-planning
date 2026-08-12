"""Best--second-best *retained-candidate* calendar audit helpers.

The myopic policy is intentionally excluded from ranked-calendar gaps: it is
constructed week-by-week, not selected from a set of complete calendars.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .boundaries import _calendar_hash, compare_schedule_maps
from .policy import solve_dynamic_category


RANKED_POLICY_SYSTEMS = {"piN": "naive_schedule_system", "piD": "schedule_system"}


def transition_neighborhoods(
    schedules: Mapping[tuple[float, int], Mapping[str, Mapping[str, Sequence[float]]]],
    reimbursement_grid: Sequence[float],
    capacities: Sequence[int],
) -> dict[tuple[float, int], str]:
    """Map each grid point to policies switching into or out of that point."""
    labels = {"piM": "myopic", "piN": "naive_dynamic", "piD": "dynamic"}
    output: dict[tuple[float, int], set[str]] = {}
    grid = [round(float(value), 8) for value in reimbursement_grid]
    for capacity in capacities:
        for before, after in zip(grid, grid[1:]):
            before_key, after_key = (before, int(capacity)), (after, int(capacity))
            for label, schedule_key in labels.items():
                if compare_schedule_maps(
                    schedules[before_key][schedule_key], schedules[after_key][schedule_key]
                )["changed_cells"]:
                    output.setdefault(before_key, set()).add(label)
                    output.setdefault(after_key, set()).add(label)
    return {key: "|".join(sorted(value)) for key, value in output.items()}


def relative_gap(best_value: float, absolute_gap: float) -> float:
    """Scale a non-negative gap by the magnitude of the selected value."""
    return absolute_gap / max(abs(best_value), 1.0)


def audit_ranked_calendars(
    artifact: Mapping[str, Any],
    *,
    near_tie_absolute: float = 1e-6,
    near_tie_relative: float = 1e-8,
    time_limit_seconds: float | None = None,
) -> pd.DataFrame:
    """Rank retained candidates, not the complete raw feasible-calendar set."""
    grid = np.asarray(artifact["reimbursement_grid"], dtype=float)
    capacities = [int(value) for value in artifact["capacities"]]
    schedules = artifact["three_policy_schedules"]
    neighborhoods = transition_neighborhoods(schedules, grid, capacities)
    rows: list[dict[str, object]] = []

    for capacity in capacities:
        for share in grid:
            key = (round(float(share), 8), capacity)
            switches = neighborhoods.get(key, "")
            for policy, system_key in RANKED_POLICY_SYSTEMS.items():
                solution = solve_dynamic_category(
                    artifact[system_key],
                    alpha=float(share),
                    capacity=capacity,
                    compute_second_best=True,
                    time_limit_seconds=time_limit_seconds,
                )
                saved_schedule_key = (
                    "naive_dynamic" if policy == "piN" else "dynamic"
                )
                if _calendar_hash(solution["schedule_map"]) != _calendar_hash(
                    schedules[key][saved_schedule_key]
                ):
                    raise AssertionError(
                        "Re-solved best calendar does not match the saved policy "
                        f"calendar for {policy}, lambda={share}, B={capacity}."
                    )
                best = float(solution["best_value"])
                second = float(solution["second_best_value"])
                absolute = float(solution["best_second_gap"])
                relative = relative_gap(best, absolute) if np.isfinite(absolute) else np.nan
                rows.append({
                    "reimbursement_share": float(share),
                    "capacity": capacity,
                    "policy": policy,
                    "ranking_basis": (
                        "full-displacement objective" if policy == "piD"
                        else "displacement-naive objective"
                    ),
                    "ranking_scope": "retained candidates; not all raw feasible calendars",
                    "best_retained_candidate_value": best,
                    "second_best_retained_candidate_value": second,
                    "retained_candidate_absolute_gap": absolute,
                    "retained_candidate_relative_gap": relative,
                    "best_calendar_hash": _calendar_hash(solution["schedule_map"]),
                    "second_best_calendar_hash": (
                        _calendar_hash(solution["second_schedule_map"])
                        if solution["second_schedule_map"] is not None else ""
                    ),
                    "near_tie": bool(
                        np.isfinite(absolute)
                        and (absolute <= near_tie_absolute or relative <= near_tie_relative)
                    ),
                    "transition_neighborhood": bool(switches),
                    "transition_policies": switches,
                    "solver_message": solution["solver_message"],
                })

            # πM is not an optimization over complete candidate calendars, so
            # a "second-best calendar" has no coherent definition here.
            rows.append({
                "reimbursement_share": float(share),
                "capacity": capacity,
                "policy": "piM",
                "ranking_basis": "not applicable: sequential myopic decisions",
                "ranking_scope": "not applicable: sequential myopic decisions",
                "best_retained_candidate_value": np.nan,
                "second_best_retained_candidate_value": np.nan,
                "retained_candidate_absolute_gap": np.nan,
                "retained_candidate_relative_gap": np.nan,
                "best_calendar_hash": _calendar_hash(schedules[key]["myopic"]),
                "second_best_calendar_hash": "",
                "near_tie": False,
                "transition_neighborhood": bool(switches),
                "transition_policies": switches,
                "solver_message": "not applicable",
            })

    return pd.DataFrame(rows).sort_values(
        ["capacity", "reimbursement_share", "policy"]
    ).reset_index(drop=True)
