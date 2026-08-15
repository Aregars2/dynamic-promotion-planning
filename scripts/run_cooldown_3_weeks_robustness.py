"""Reoptimize the frozen policy problem with a three-week promotion cooldown.

This is a post-specification feasible-set robustness check.  It preserves the
empirical-Bayes demand profiles, behavioral draws, action support, economics,
and all policy algorithms; only ``PlanningSpec.cooldown`` and its implied
promotion cap change.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dynamic_promotion_planning.policy import (
    build_schedule_system,
    evaluate_schedule_map,
    load_pickle,
    maximum_feasible_promotions,
    run_three_policy_grid,
    save_pickle,
)
from scripts.run_task7_robustness import summarize


VERSION = "empirical_bayes_price_consistent"
SPECIFICATION = "cooldown_3_weeks"
MAIN_ARTIFACT = ROOT / "artifacts" / "policy" / VERSION / "policy_optimization.pkl"
TABLE_DIR = ROOT / "results" / VERSION / "robustness" / SPECIFICATION
ARTIFACT_DIR = ROOT / "artifacts" / "robustness" / VERSION / SPECIFICATION


def _prefix_profiles(profiles: dict, weeks: int) -> dict:
    """Keep only the planning horizon when calculating the own denominator."""
    return {
        upc: {key: np.asarray(value)[:weeks] for key, value in profile.items()}
        for upc, profile in profiles.items()
    }


def run_cooldown_3_weeks_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run πM, πN, and πD with ``P,0,0,0,P`` as the closest spacing."""
    started = time.monotonic()
    if not MAIN_ARTIFACT.is_file():
        raise FileNotFoundError(
            f"Missing frozen main artifact: {MAIN_ARTIFACT}. Run Notebook 06 first."
        )
    main = load_pickle(MAIN_ARTIFACT)
    base_planning = main["schedule_system"]["planning"]
    planning = replace(
        base_planning,
        cooldown=3,
        max_promotions=maximum_feasible_promotions(base_planning.decision_horizon, 3),
    )
    if planning.max_promotions != 3:
        raise AssertionError("A 12-week horizon with a three-week cooldown must allow at most three promotions.")

    draws = main["draws_by_product"]
    profiles = main["weekly_profiles"]
    actions = main["action_sets"]
    grid = list(main["reimbursement_grid"])
    capacities = list(main["capacities"])

    print("Cooldown-3 robustness: constructing displacement-aware candidates...", flush=True)
    dynamic_system = build_schedule_system(draws, profiles, actions, planning, grid)
    print("Cooldown-3 robustness: constructing displacement-naive candidates...", flush=True)
    naive_system = build_schedule_system(
        draws,
        profiles,
        actions,
        planning,
        grid,
        add_new_promotion_displacement=False,
    )
    print("Cooldown-3 robustness: optimizing πM, πN, and πD...", flush=True)
    run = run_three_policy_grid(
        dynamic_system,
        naive_system,
        draws,
        profiles,
        actions,
        grid,
        capacities,
        compute_second_best=False,
    )

    peak_summary, full_grid = summarize(
        SPECIFICATION, run["results"], run["schedules"], capacities
    )
    planning_12 = replace(planning, washout_horizon=0)
    profiles_12 = _prefix_profiles(profiles, planning_12.evaluation_horizon)
    denominators = []
    for row in full_grid.itertuples(index=False):
        key = (round(float(row.reimbursement_share), 8), int(row.capacity))
        myopic = evaluate_schedule_map(
            run["schedules"][key]["myopic"],
            draws,
            profiles_12,
            planning_12,
            float(row.reimbursement_share),
        )["total_profit"]
        if not np.isfinite(myopic) or myopic <= 0:
            raise AssertionError("Cooldown-3 myopic 12-week denominator must be finite and positive.")
        denominators.append(float(myopic))
    full_grid["myopic_planning_profit_12w"] = denominators
    full_grid["delta_plan_pct_myopic_planning_12w"] = (
        100.0 * full_grid["delta_plan"] / full_grid["myopic_planning_profit_12w"]
    )
    full_grid["delta_disp_pct_myopic_planning_12w"] = (
        100.0 * full_grid["delta_disp"] / full_grid["myopic_planning_profit_12w"]
    )
    full_grid["delta_total_pct_myopic_planning_12w"] = (
        100.0 * full_grid["delta_total"] / full_grid["myopic_planning_profit_12w"]
    )
    if not np.allclose(
        full_grid["delta_plan_pct_myopic_planning_12w"]
        + full_grid["delta_disp_pct_myopic_planning_12w"],
        full_grid["delta_total_pct_myopic_planning_12w"],
        atol=1e-10,
    ):
        raise AssertionError("Cooldown-3 normalized three-policy components do not add up.")

    peaks = (
        full_grid.loc[full_grid.groupby("capacity", observed=True)["delta_total"].idxmax()]
        .sort_values("capacity")
        .reset_index(drop=True)
    )
    for peak in peaks.itertuples(index=False):
        group = full_grid.loc[full_grid.capacity.eq(peak.capacity)]
        if not np.isclose(peak.delta_total, group.delta_total.max()):
            raise AssertionError(f"Cooldown-3 B={peak.capacity}: reported peak is not the maximizer.")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    full_grid.to_csv(TABLE_DIR / "cooldown_3_weeks_full_grid.csv", index=False)
    peak_summary.to_csv(TABLE_DIR / "cooldown_3_weeks_transition_summary.csv", index=False)
    peaks.to_csv(TABLE_DIR / "cooldown_3_weeks_peak_summary.csv", index=False)
    save_pickle(
        {
            "run": run,
            "planning": planning,
            "source_main_artifact": str(MAIN_ARTIFACT),
            "robustness_change": "cooldown=3; max_promotions=3",
        },
        ARTIFACT_DIR / "policy_optimization.pkl",
    )
    elapsed = time.monotonic() - started
    print(peaks[["capacity", "reimbursement_share", "delta_total", "delta_total_pct_myopic_planning_12w"]].to_string(index=False))
    print(f"Cooldown-3 robustness completed in {elapsed:.1f}s.", flush=True)
    return full_grid, peaks


if __name__ == "__main__":
    run_cooldown_3_weeks_robustness()
