from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_promotion_planning.boundaries import (
    build_boundary_transition_table,
    build_global_transition_table,
    build_three_policy_transition_table,
    compare_schedule_maps,
    mark_schedule_boundaries,
    validate_grid_overlap,
)


def test_grid_overlap_accepts_numerically_identical_results() -> None:
    coarse = pd.DataFrame(
        {
            "alpha": [1.0, 1.5, 2.0],
            "dynamic_profit": [10.0, 11.0, 12.0],
            "myopic_profit": [9.0, 9.5, 10.0],
            "vdo": [1.0, 1.5, 2.0],
        }
    )
    fine = pd.DataFrame(
        {
            "alpha": [1.0, 1.25, 1.5, 1.75, 2.0],
            "dynamic_profit": [10.0, 10.5, 11.0, 11.5, 12.0],
            "myopic_profit": [9.0, 9.25, 9.5, 9.75, 10.0],
            "vdo": [1.0, 1.25, 1.5, 1.75, 2.0],
        }
    )
    comparison = validate_grid_overlap(fine, coarse, tolerance=1e-12)
    assert len(comparison) == 3
    assert comparison.filter(like="_difference").abs().to_numpy().max() == 0.0


def test_grid_overlap_raises_on_material_mismatch() -> None:
    coarse = pd.DataFrame(
        {
            "alpha": [1.0],
            "dynamic_profit": [10.0],
            "myopic_profit": [9.0],
            "vdo": [1.0],
        }
    )
    fine = coarse.copy()
    fine.loc[0, "vdo"] = 1.1
    with pytest.raises(AssertionError):
        validate_grid_overlap(fine, coarse, tolerance=1e-5)


def test_schedule_comparison_distinguishes_status_and_depth_changes() -> None:
    previous = {
        "A": np.array([0.1, 0.0, 0.0]),
        "B": np.array([0.0, 0.1, 0.0]),
    }
    current = {
        "A": np.array([0.2, 0.0, 0.0]),
        "B": np.array([0.0, 0.0, 0.1]),
    }
    result = compare_schedule_maps(previous, current)
    assert result["changed_product_count"] == 2
    assert result["changed_cells"] == 3
    assert result["depth_changes"] == 1
    assert result["status_changes"] == 2


def test_boundary_transition_classifies_myopic_only_switch() -> None:
    results = pd.DataFrame(
        {
            "alpha": [1.0, 1.1],
            "vdo": [1.0, 2.0],
            "vdo_percent": [0.1, 0.2],
            "dynamic_schedule_signature": ["d0", "d0"],
            "myopic_schedule_signature": ["m0", "m1"],
            "dynamic_promotion_count": [1, 1],
            "myopic_promotion_count": [1, 1],
            "best_second_gap": [0.5, 0.4],
        }
    )
    schedules = {
        (1.0, 2): {
            "dynamic": {"A": np.array([0.1, 0.0])},
            "myopic": {"A": np.array([0.1, 0.0])},
        },
        (1.1, 2): {
            "dynamic": {"A": np.array([0.1, 0.0])},
            "myopic": {"A": np.array([0.0, 0.1])},
        },
    }
    marked, transitions = build_boundary_transition_table(
        results,
        schedules,
        capacity=2,
    )
    assert not bool(marked.loc[0, "dynamic_boundary"])
    assert not bool(marked.loc[0, "myopic_boundary"])
    assert transitions.loc[0, "boundary_type"] == "myopic_only"
    assert transitions.loc[0, "myopic_status_changes"] == 2


def test_first_grid_point_is_not_marked_as_boundary() -> None:
    frame = pd.DataFrame(
        {
            "alpha": [1.0],
            "dynamic_schedule_signature": ["d0"],
            "myopic_schedule_signature": ["m0"],
        }
    )
    marked = mark_schedule_boundaries(frame)
    assert not bool(marked.loc[0, "dynamic_boundary"])
    assert not bool(marked.loc[0, "myopic_boundary"])


def test_global_transition_table_keeps_capacity_groups_separate() -> None:
    results = pd.DataFrame(
        {
            "alpha": [1.0, 1.1, 1.0, 1.1],
            "capacity": [1, 1, 2, 2],
            "vdo": [0.0, 1.0, 0.0, 0.0],
            "vdo_percent": [0.0, 0.1, 0.0, 0.0],
            "dynamic_schedule_signature": ["d0", "d0", "d0", "d0"],
            "myopic_schedule_signature": ["m0", "m1", "m0", "m0"],
            "dynamic_promotion_count": [0, 0, 0, 0],
            "myopic_promotion_count": [0, 1, 0, 0],
            "best_second_gap": [1.0, 1.0, 1.0, 1.0],
        }
    )
    schedules = {
        (1.0, 1): {"dynamic": {"A": np.zeros(1)}, "myopic": {"A": np.zeros(1)}},
        (1.1, 1): {"dynamic": {"A": np.zeros(1)}, "myopic": {"A": np.array([0.1])}},
        (1.0, 2): {"dynamic": {"A": np.zeros(1)}, "myopic": {"A": np.zeros(1)}},
        (1.1, 2): {"dynamic": {"A": np.zeros(1)}, "myopic": {"A": np.zeros(1)}},
    }
    marked, transitions = build_global_transition_table(results, schedules)
    assert marked.groupby("capacity")["myopic_boundary"].first().eq(False).all()
    assert len(transitions) == 1
    assert int(transitions.loc[0, "capacity"]) == 1


def test_three_policy_transition_table_reports_each_switching_calendar() -> None:
    results = pd.DataFrame(
        {
            "capacity": [2, 2],
            "reimbursement_share": [0.10, 0.11],
            "delta_plan": [1.0, 1.2],
            "delta_disp": [0.5, 0.4],
            "delta_total": [1.5, 1.6],
        }
    )
    schedules = {
        (0.10, 2): {
            "myopic": {"A": np.array([0.0, 0.1])},
            "naive_dynamic": {"A": np.array([0.1, 0.0])},
            "dynamic": {"A": np.array([0.1, 0.0])},
        },
        (0.11, 2): {
            "myopic": {"A": np.array([0.0, 0.1])},
            "naive_dynamic": {"A": np.array([0.0, 0.1])},
            "dynamic": {"A": np.array([0.0, 0.0])},
        },
    }
    transitions = build_three_policy_transition_table(results, schedules, capacity=2)
    assert transitions["switching_policy"].tolist() == ["piN", "piD"]
    assert transitions["changed_product_week_decisions"].tolist() == [2, 1]
    assert transitions["calendar_hash_before"].str.len().eq(16).all()
