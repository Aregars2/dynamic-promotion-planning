"""Boundary detection and coarse/fine-grid validation helpers."""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _calendar_hash(schedule: Mapping[str, Sequence[float]]) -> str:
    """Stable compact hash for a product-week promotion calendar."""
    digest = hashlib.sha256()
    for upc in sorted(schedule):
        digest.update(str(upc).encode("utf-8"))
        digest.update(np.asarray(schedule[upc], dtype=np.float64).round(8).tobytes())
    return digest.hexdigest()[:16]


def changed_products(
    previous_schedule: dict[str, np.ndarray],
    current_schedule: dict[str, np.ndarray],
) -> tuple[list[str], int]:
    products = sorted(
        set(previous_schedule)
        | set(current_schedule)
    )

    changed = []
    changed_cells = 0

    for upc in products:
        previous = np.asarray(
            previous_schedule[upc],
            dtype=float,
        )
        current = np.asarray(
            current_schedule[upc],
            dtype=float,
        )

        differences = ~np.isclose(
            previous,
            current,
        )

        if differences.any():
            changed.append(upc)
            changed_cells += int(
                differences.sum()
            )

    return changed, changed_cells

def compare_schedule_maps(
    previous: Mapping[
        str,
        Sequence[float],
    ],
    current: Mapping[
        str,
        Sequence[float],
    ],
) -> dict[str, object]:
    all_products = sorted(
        set(previous)
        | set(current)
    )

    changed_products = []
    changed_cells = 0
    status_changes = 0
    depth_changes = 0

    for upc in all_products:
        previous_values = np.asarray(
            previous[upc],
            dtype=float,
        )
        current_values = np.asarray(
            current[upc],
            dtype=float,
        )

        changed = ~np.isclose(
            previous_values,
            current_values,
        )

        if changed.any():
            changed_products.append(
                str(upc)
            )
            changed_cells += int(
                changed.sum()
            )

        previous_positive = (
            previous_values > 0
        )
        current_positive = (
            current_values > 0
        )

        status_changes += int(
            np.sum(
                previous_positive
                != current_positive
            )
        )

        depth_changes += int(
            np.sum(
                previous_positive
                & current_positive
                & changed
            )
        )

    return {
        "changed_products": "|".join(
            changed_products
        ),
        "changed_product_count": len(
            changed_products
        ),
        "changed_cells": changed_cells,
        "status_changes": status_changes,
        "depth_changes": depth_changes,
    }

def validate_grid_overlap(
    fine_results: pd.DataFrame,
    coarse_results: pd.DataFrame,
    *,
    tolerance: float = 1e-5,
) -> pd.DataFrame:
    """Compare fine- and coarse-grid values at common funding parameters."""
    overlap = sorted(
        set(np.round(fine_results["alpha"], 8)).intersection(
            set(np.round(coarse_results["alpha"], 8))
        )
    )
    comparison = (
        fine_results.loc[
            fine_results["alpha"].round(8).isin(overlap),
            ["alpha", "dynamic_profit", "myopic_profit", "vdo"],
        ]
        .merge(
            coarse_results.loc[
                coarse_results["alpha"].round(8).isin(overlap),
                ["alpha", "dynamic_profit", "myopic_profit", "vdo"],
            ],
            on="alpha",
            suffixes=("_fine", "_coarse"),
            validate="one_to_one",
        )
    )
    for column in ["dynamic_profit", "myopic_profit", "vdo"]:
        comparison[f"{column}_difference"] = (
            comparison[f"{column}_fine"] - comparison[f"{column}_coarse"]
        )
    difference_columns = [
        column for column in comparison if column.endswith("_difference")
    ]
    if difference_columns and (
        comparison[difference_columns].abs().to_numpy().max() > tolerance
    ):
        raise AssertionError(
            "Fine-grid values do not reproduce the coarse grid at overlap points."
        )
    return comparison


def mark_schedule_boundaries(results: pd.DataFrame) -> pd.DataFrame:
    """Mark adjacent changes in dynamic and myopic schedule signatures."""
    output = results.sort_values("alpha").reset_index(drop=True).copy()
    output["dynamic_boundary"] = output["dynamic_schedule_signature"].ne(
        output["dynamic_schedule_signature"].shift()
    )
    output["myopic_boundary"] = output["myopic_schedule_signature"].ne(
        output["myopic_schedule_signature"].shift()
    )
    if not output.empty:
        output.loc[0, ["dynamic_boundary", "myopic_boundary"]] = False
    return output


def build_policy_switch_table(
    policy_results: pd.DataFrame,
    schedules: Mapping[tuple[float, int], Mapping[str, Mapping[str, Sequence[float]]]],
) -> pd.DataFrame:
    """Summarize adjacent coarse-grid schedule changes by capacity."""
    rows: list[dict[str, object]] = []
    for capacity, group in policy_results.groupby("capacity", observed=True):
        previous_row = None
        for row in group.sort_values("alpha").itertuples(index=False):
            current_key = (round(float(row.alpha), 8), int(capacity))
            if previous_row is not None:
                previous_key = (
                    round(float(previous_row.alpha), 8),
                    int(capacity),
                )
                dynamic_products, dynamic_cells = changed_products(
                    schedules[previous_key]["dynamic"],
                    schedules[current_key]["dynamic"],
                )
                myopic_products, myopic_cells = changed_products(
                    schedules[previous_key]["myopic"],
                    schedules[current_key]["myopic"],
                )
                if dynamic_cells > 0 or myopic_cells > 0:
                    rows.append(
                        {
                            "capacity": int(capacity),
                            "alpha_previous": float(previous_row.alpha),
                            "alpha_current": float(row.alpha),
                            "vdo_previous": float(previous_row.vdo),
                            "vdo_current": float(row.vdo),
                            "vdo_change": float(row.vdo - previous_row.vdo),
                            "dynamic_changed_products": "|".join(dynamic_products),
                            "dynamic_changed_cells": int(dynamic_cells),
                            "myopic_changed_products": "|".join(myopic_products),
                            "myopic_changed_cells": int(myopic_cells),
                        }
                    )
            previous_row = row
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            "vdo_change",
            key=lambda values: values.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_three_policy_transition_table(
    policy_results: pd.DataFrame,
    schedules: Mapping[tuple[float, int], Mapping[str, Mapping[str, Sequence[float]]]],
    *,
    capacity: int,
) -> pd.DataFrame:
    """Return one row per πM, πN, or πD calendar change on an adjacent grid.

    Values labelled ``before`` and ``after`` are the three additive value
    components at the two reimbursement shares bracketing a switch.
    """
    required = {"reimbursement_share", "delta_plan", "delta_disp", "delta_total"}
    missing = required.difference(policy_results.columns)
    if missing:
        raise KeyError(f"Missing three-policy result columns: {sorted(missing)}")
    policy_keys = {"piM": "myopic", "piN": "naive_dynamic", "piD": "dynamic"}
    rows: list[dict[str, object]] = []
    group = policy_results.loc[policy_results["capacity"].eq(capacity)].sort_values(
        "reimbursement_share"
    )
    for position in range(1, len(group)):
        before = group.iloc[position - 1]
        after = group.iloc[position]
        before_key = (round(float(before["reimbursement_share"]), 8), int(capacity))
        after_key = (round(float(after["reimbursement_share"]), 8), int(capacity))
        for label, schedule_key in policy_keys.items():
            old_schedule = schedules[before_key][schedule_key]
            new_schedule = schedules[after_key][schedule_key]
            change = compare_schedule_maps(old_schedule, new_schedule)
            if not change["changed_cells"]:
                continue
            rows.append({
                "reimbursement_share_previous": float(before["reimbursement_share"]),
                "reimbursement_share": float(after["reimbursement_share"]),
                "switching_policy": label,
                "calendar_hash_before": _calendar_hash(old_schedule),
                "calendar_hash_after": _calendar_hash(new_schedule),
                "changed_product_week_decisions": int(change["changed_cells"]),
                "changed_products": change["changed_products"],
                "delta_plan_before": float(before["delta_plan"]),
                "delta_disp_before": float(before["delta_disp"]),
                "delta_total_before": float(before["delta_total"]),
                "delta_plan_after": float(after["delta_plan"]),
                "delta_disp_after": float(after["delta_disp"]),
                "delta_total_after": float(after["delta_total"]),
            })
    return pd.DataFrame(rows)


def build_boundary_transition_table(
    local_results: pd.DataFrame,
    local_schedules: Mapping[
        tuple[float, int],
        Mapping[str, Mapping[str, Sequence[float]]],
    ],
    capacity: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify adjacent local schedule transitions for both policies."""
    marked = mark_schedule_boundaries(local_results)
    rows: list[dict[str, object]] = []

    for position in range(1, len(marked)):
        previous_row = marked.iloc[position - 1]
        current_row = marked.iloc[position]
        dynamic_boundary = bool(current_row["dynamic_boundary"])
        myopic_boundary = bool(current_row["myopic_boundary"])
        if not (dynamic_boundary or myopic_boundary):
            continue

        previous_key = (round(float(previous_row["alpha"]), 8), int(capacity))
        current_key = (round(float(current_row["alpha"]), 8), int(capacity))
        dynamic_change = compare_schedule_maps(
            local_schedules[previous_key]["dynamic"],
            local_schedules[current_key]["dynamic"],
        )
        myopic_change = compare_schedule_maps(
            local_schedules[previous_key]["myopic"],
            local_schedules[current_key]["myopic"],
        )

        boundary_type = (
            "both"
            if dynamic_boundary and myopic_boundary
            else "dynamic_only"
            if dynamic_boundary
            else "myopic_only"
        )
        rows.append(
            {
                "capacity": int(capacity),
                "alpha_previous": float(previous_row["alpha"]),
                "alpha_current": float(current_row["alpha"]),
                "boundary_type": boundary_type,
                "vdo_previous": float(previous_row["vdo"]),
                "vdo_current": float(current_row["vdo"]),
                "vdo_change": float(
                    current_row["vdo"] - previous_row["vdo"]
                ),
                "vdo_percent_current": float(current_row["vdo_percent"]),
                "dynamic_promotion_count_previous": int(
                    previous_row["dynamic_promotion_count"]
                ),
                "dynamic_promotion_count_current": int(
                    current_row["dynamic_promotion_count"]
                ),
                "myopic_promotion_count_previous": int(
                    previous_row["myopic_promotion_count"]
                ),
                "myopic_promotion_count_current": int(
                    current_row["myopic_promotion_count"]
                ),
                "best_second_gap_current": float(
                    current_row["best_second_gap"]
                ),
                **{
                    f"dynamic_{key}": value
                    for key, value in dynamic_change.items()
                },
                **{
                    f"myopic_{key}": value
                    for key, value in myopic_change.items()
                },
            }
        )

    transitions = pd.DataFrame(rows)
    if not transitions.empty:
        transitions["absolute_vdo_change"] = transitions["vdo_change"].abs()
        transitions = transitions.sort_values(
            "absolute_vdo_change",
            ascending=False,
        ).reset_index(drop=True)
    return marked, transitions


def build_global_transition_table(
    policy_results: pd.DataFrame,
    schedules: Mapping[
        tuple[float, int], Mapping[str, Mapping[str, Sequence[float]]]
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify schedule transitions across all capacities and funding values."""
    marked_frames: list[pd.DataFrame] = []
    transition_frames: list[pd.DataFrame] = []

    for capacity, group in policy_results.groupby("capacity", observed=True):
        marked, transitions = build_boundary_transition_table(
            local_results=group.sort_values("alpha").reset_index(drop=True),
            local_schedules=schedules,
            capacity=int(capacity),
        )
        marked_frames.append(marked)
        if not transitions.empty:
            transition_frames.append(transitions)

    marked_results = (
        pd.concat(marked_frames, ignore_index=True)
        .sort_values(["capacity", "alpha"])
        .reset_index(drop=True)
    )
    transitions = (
        pd.concat(transition_frames, ignore_index=True)
        if transition_frames
        else pd.DataFrame()
    )
    if not transitions.empty:
        transitions = transitions.sort_values(
            ["capacity", "alpha_current", "boundary_type"]
        ).reset_index(drop=True)
    return marked_results, transitions
