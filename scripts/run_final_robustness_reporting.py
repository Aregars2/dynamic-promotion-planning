"""Post-specification reporting and fixed-calendar behavioral uncertainty.

This script is deliberately read-only with respect to the frozen main policy
artifact. It writes only below ``results/empirical_bayes_price_consistent/robustness``.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_promotion_planning.policy import evaluate_schedule_map, load_pickle


EB_CALIBRATION = ROOT / "artifacts" / "calibration" / "empirical_bayes"
EB_POLICY = ROOT / "artifacts" / "policy" / "empirical_bayes_price_consistent"
OUT = ROOT / "results" / "empirical_bayes_price_consistent" / "robustness"


def _quantile(values: np.ndarray, probabilities: list[float]) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=float), probabilities)


def same_horizon_normalization(artifact: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report both planning-profit and 48-week-myopic-profit normalizations."""
    planning = artifact["schedule_system"]["planning"]
    rows: list[dict[str, float | int]] = []
    for record in artifact["policy_results"].itertuples(index=False):
        key = (round(float(record.reimbursement_share), 8), int(record.capacity))
        myopic = artifact["three_policy_schedules"][key]["myopic"]
        detail = evaluate_schedule_map(
            myopic, artifact["draws_by_product"], artifact["weekly_profiles"],
            planning, float(record.reimbursement_share),
        )
        horizon = detail["weekly_profit"].loc[lambda x: x["decision_week"], "profit"].sum()
        full = float(detail["total_profit"])
        if horizon == 0 or full == 0:
            raise AssertionError("Myopic normalization denominator is zero.")
        row = {
            "reimbursement_share": float(record.reimbursement_share),
            "capacity": int(record.capacity),
            "myopic_planning_profit_12w": float(horizon),
            "myopic_profit_48w": full,
        }
        for component in ("plan", "disp", "total"):
            value = float(getattr(record, f"delta_{component}"))
            row[f"delta_{component}"] = value
            row[f"delta_{component}_pct_myopic_planning_12w"] = 100.0 * value / horizon
            row[f"delta_{component}_pct_myopic_same_horizon_48w"] = 100.0 * value / full
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values(["capacity", "reimbursement_share"])
    for suffix in ("pct_myopic_planning_12w", "pct_myopic_same_horizon_48w"):
        if not np.allclose(grid[f"delta_plan_{suffix}"] + grid[f"delta_disp_{suffix}"], grid[f"delta_total_{suffix}"], atol=1e-10):
            raise AssertionError(f"Normalized decomposition fails for {suffix}.")
    peaks = grid.loc[grid.groupby("capacity", observed=True)["delta_total"].idxmax()].copy()
    return grid.reset_index(drop=True), peaks.sort_values("capacity").reset_index(drop=True)


def _parent_policy_values(
    schedule: dict[str, np.ndarray], artifact: dict, share: float, parent_ids: np.ndarray
) -> np.ndarray:
    """Evaluate one fixed calendar by coherent bootstrap parent, conditioning on persistence children."""
    raw = pd.read_pickle(EB_CALIBRATION / "product_behavioral_draws.pkl").copy()
    raw["upc"] = raw["upc"].astype(str)
    if set(parent_ids) != set(range(1000)):
        raise AssertionError("Expected coherent parent bootstrap IDs 0,...,999.")
    output = np.zeros(len(parent_ids), dtype=float)
    planning = artifact["schedule_system"]["planning"]
    for upc in sorted(artifact["draws_by_product"]):
        product = raw.loc[raw["upc"].eq(str(upc))].copy()
        child_counts = product.groupby("bootstrap_id", observed=True).size()
        if not child_counts.index.equals(pd.Index(parent_ids)):
            raise AssertionError(f"Incomplete bootstrap-parent coverage for UPC {upc}.")
        for position, parent in enumerate(parent_ids):
            children = product.loc[product["bootstrap_id"].eq(parent)]
            weights = children["draw_weight"].to_numpy(dtype=float, copy=True)
            weights /= weights.sum()
            draws = {
                "epsilon": children["price_elasticity"].to_numpy(float),
                "gamma": children["promotion_lift_log"].to_numpy(float),
                "psi": children["displacement_strength"].to_numpy(float),
                "r": children["inventory_persistence"].to_numpy(float),
                "weights": weights,
                "base_demand": children["base_demand"].to_numpy(float),
                "regular_price": children["regular_price"].to_numpy(float),
                "unit_cost": children["unit_cost"].to_numpy(float),
            }
            output[position] += evaluate_schedule_map(
                {str(upc): schedule[str(upc)]}, {str(upc): draws},
                {str(upc): artifact["weekly_profiles"][str(upc)]}, planning, share,
            )["total_profit"]
    return output


def fixed_calendar_uncertainty(artifact: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired contrasts at each capacity-specific main peak; no reoptimization."""
    parent_ids = np.arange(1000, dtype=int)
    main = artifact["policy_results"]
    peak_rows = main.loc[main.groupby("capacity", observed=True)["delta_total"].idxmax()]
    long_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for peak in peak_rows.itertuples(index=False):
        share, capacity = float(peak.reimbursement_share), int(peak.capacity)
        calendars = artifact["three_policy_schedules"][(round(share, 8), capacity)]
        values = {
            "piM": _parent_policy_values(calendars["myopic"], artifact, share, parent_ids),
            "piN": _parent_policy_values(calendars["naive_dynamic"], artifact, share, parent_ids),
            "piD": _parent_policy_values(calendars["dynamic"], artifact, share, parent_ids),
        }
        contrasts = {
            "delta_plan": values["piN"] - values["piM"],
            "delta_disp": values["piD"] - values["piN"],
            "delta_total": values["piD"] - values["piM"],
        }
        if not np.allclose(contrasts["delta_plan"] + contrasts["delta_disp"], contrasts["delta_total"], atol=1e-8):
            raise AssertionError("Parent-level paired contrasts do not add up.")
        if not np.isfinite(values["piM"]).all() or (values["piM"] <= 0).any():
            raise AssertionError("Parent-level 48-week myopic values must be finite and positive.")
        total_pct = 100.0 * contrasts["delta_total"] / values["piM"]
        for parent_position, parent in enumerate(parent_ids):
            for component, values_by_parent in contrasts.items():
                long_rows.append({"capacity": capacity, "reimbursement_share": share, "bootstrap_id": int(parent), "component": component, "contrast": float(values_by_parent[parent_position])})
            long_rows.append({"capacity": capacity, "reimbursement_share": share, "bootstrap_id": int(parent), "component": "delta_total_pct_myopic_48w", "contrast": float(total_pct[parent_position])})
        for component, values_by_parent in contrasts.items():
            p05, p50, p95 = _quantile(values_by_parent, [0.05, 0.50, 0.95])
            summary_rows.append({
                "capacity": capacity, "reimbursement_share": share, "component": component,
                "p05": float(p05), "p50": float(p50), "p95": float(p95),
                "label": "behavioral-parameter uncertainty conditional on the selected calendars",
                "bootstrap_parents": len(parent_ids),
            })
        p05, p50, p95 = _quantile(total_pct, [0.05, 0.50, 0.95])
        summary_rows.append({
            "capacity": capacity, "reimbursement_share": share,
            "component": "delta_total_pct_myopic_48w",
            "p05": float(p05), "p50": float(p50), "p95": float(p95),
            "label": "behavioral-parameter uncertainty conditional on the selected calendars",
            "bootstrap_parents": len(parent_ids),
        })
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def run_reporting_robustness() -> dict[str, pd.DataFrame]:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    artifact_path = EB_POLICY / "policy_optimization.pkl"
    artifact = load_pickle(artifact_path)
    grid, peaks = same_horizon_normalization(artifact)
    grid.to_csv(OUT / "same_horizon_normalization_full_grid.csv", index=False)
    peaks.to_csv(OUT / "same_horizon_normalization_peak_summary.csv", index=False)
    long, summary = fixed_calendar_uncertainty(artifact)
    long.to_csv(OUT / "fixed_calendar_behavioral_uncertainty_peak_draws.csv", index=False)
    summary.to_csv(OUT / "fixed_calendar_behavioral_uncertainty_peak_summary.csv", index=False)
    runtime_seconds = time.monotonic() - started
    return {
        "same_horizon_grid": grid,
        "same_horizon_peaks": peaks,
        "fixed_calendar_parent_contrasts": long,
        "fixed_calendar_summary": summary,
        "runtime_seconds": runtime_seconds,
    }


def main() -> None:
    output = run_reporting_robustness()
    print(f"Completed reporting robustness in {output['runtime_seconds']:.1f}s.")


if __name__ == "__main__":
    main()
