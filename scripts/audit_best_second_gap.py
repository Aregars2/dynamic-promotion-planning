"""Audit best versus second-best retained candidates on the 0.01 grid.

This is not a literal second-best feasible-calendar diagnostic: schedules that
lose within the same occupancy mask are safely pruned for finding an optimum,
but can still be a raw runner-up after the optimum is excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dynamic_promotion_planning.gap_audit import audit_ranked_calendars
from dynamic_promotion_planning.policy import load_pickle, save_pickle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--near-tie-absolute", type=float, default=1e-6)
    parser.add_argument("--near-tie-relative", type=float, default=1e-8)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    args = parser.parse_args()

    artifact_path = PROJECT_ROOT / "artifacts" / "policy" / "policy_optimization.pkl"
    artifact = load_pickle(artifact_path)
    grid = np.asarray(artifact["schedule_system"]["alpha_grid"], dtype=float)
    expected = np.round(np.arange(0.0, 1.001, 0.01), 8)
    if not np.array_equal(np.round(grid, 8), expected):
        raise AssertionError("The candidate schedules were not pruned on the 0.01 grid.")

    audit = audit_ranked_calendars(
        artifact,
        near_tie_absolute=args.near_tie_absolute,
        near_tie_relative=args.near_tie_relative,
        time_limit_seconds=args.time_limit_seconds,
    )
    table_path = PROJECT_ROOT / "results" / "final" / "tables" / "best_second_gap_audit.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(table_path, index=False)
    artifact_output = PROJECT_ROOT / "artifacts" / "policy" / "best_second_gap_audit.pkl"
    save_pickle({
        "audit": audit,
        "source_artifact": str(artifact_path.name),
        "candidate_grid": grid,
        "near_tie_absolute": args.near_tie_absolute,
        "near_tie_relative": args.near_tie_relative,
    }, artifact_output)
    ranked = audit.loc[audit["policy"].isin(["piN", "piD"])]
    print(f"Saved: {table_path}")
    print(f"Saved: {artifact_output}")
    print(f"Retained-candidate rows: {len(ranked)}; near ties: {int(ranked['near_tie'].sum())}")
    print(f"Transition-neighborhood ranked rows: {int(ranked['transition_neighborhood'].sum())}")


if __name__ == "__main__":
    main()
