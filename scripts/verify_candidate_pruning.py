"""Verify same-mask candidate pruning is exact on the reported 0.01 grid."""
from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dynamic_promotion_planning.policy import load_pickle


def verify_system(label: str, system: dict) -> dict[str, object]:
    grid = np.round(np.asarray(system["alpha_grid"], dtype=float), 8)
    expected = np.round(np.arange(0.0, 1.001, 0.01), 8)
    if not np.array_equal(grid, expected):
        raise AssertionError(f"{label} was not pruned on the complete 0.01 grid.")

    checks = discarded = 0
    max_loss = 0.0
    for artifact in system["product_artifacts"].values():
        all_values = artifact["schedule_values"]
        retained = artifact["candidates"]
        discarded += len(all_values) - len(retained)
        for share in grid:
            full = all_values.assign(value=all_values["intercept"] + share * all_values["exposure"])
            kept = retained.assign(value=retained["intercept"] + share * retained["exposure"])
            full_max = full.groupby("occupancy_mask", observed=True)["value"].max()
            kept_max = kept.groupby("occupancy_mask", observed=True)["value"].max().reindex(full_max.index)
            if kept_max.isna().any():
                raise AssertionError(f"{label} is missing a retained occupancy mask at lambda={share}.")
            loss = float(np.abs((full_max - kept_max).to_numpy(dtype=float)).max())
            max_loss = max(max_loss, loss)
            checks += len(full_max)
    if max_loss > 1e-9:
        raise AssertionError(f"{label} pruning loss is {max_loss}, not zero.")
    return {
        "policy_system": label,
        "reimbursement_grid_points": len(grid),
        "products": len(system["product_artifacts"]),
        "occupancy_grid_checks": checks,
        "discarded_schedules": discarded,
        "max_full_minus_retained_optimum_loss": max_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify exact same-mask candidate pruning for a policy artifact."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "policy"
        / "empirical_bayes_price_consistent"
        / "policy_optimization.pkl",
        help="Policy artifact to audit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "empirical_bayes_price_consistent"
        / "tables"
        / "candidate_pruning_exactness.csv",
        help="CSV path for the exactness audit.",
    )
    args = parser.parse_args()

    artifact = load_pickle(args.artifact)
    output = pd.DataFrame([
        verify_system("piN", artifact["naive_schedule_system"]),
        verify_system("piD", artifact["schedule_system"]),
    ])
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(output.to_string(index=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
