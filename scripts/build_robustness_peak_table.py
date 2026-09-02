"""Build the paper-facing robustness table from executed robustness outputs."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_promotion_planning.policy import evaluate_schedule_map, load_pickle

VERSION = "empirical_bayes_price_consistent"
TABLE_DIR = ROOT / "results" / VERSION / "tables"
ROBUSTNESS_DIR = ROOT / "results" / VERSION / "robustness"
ALT_ARTIFACT = (
    ROOT / "artifacts" / "robustness" / VERSION
    / "alternative_ppml_product_promotion" / "policy_optimization.pkl"
)
TASK7_ARTIFACT = ROOT / "artifacts" / "policy" / VERSION / "task7_robustness_runs.pkl"
MAIN_ARTIFACT = ROOT / "artifacts" / "policy" / VERSION / "policy_optimization.pkl"
CAPACITIES = (1, 2, 3, 8)
LABELS = {
    "main": "Main",
    "pooled_displacement": "Pooled displacement",
    "exclude_3800001611": "Without Special K",
    "cooldown_3_weeks": "Three-week spacing",
    "alternative_product_promotion": "Alternative demand model",
}


def _prefix_profiles(profiles: dict, weeks: int) -> dict:
    return {
        upc: {key: np.asarray(value)[:weeks] for key, value in profile.items()}
        for upc, profile in profiles.items()
    }


def _assert_peak(frame: pd.DataFrame, specification: str) -> pd.DataFrame:
    rows = []
    for capacity in CAPACITIES:
        group = frame.loc[frame["capacity"].eq(capacity)].copy()
        if group.empty:
            raise AssertionError(f"{specification}: missing B={capacity}.")
        peak = group.loc[group["delta_total"].idxmax()]
        if not np.isclose(peak.delta_total, group.delta_total.max()):
            raise AssertionError(f"{specification}, B={capacity}: reported row is not the maximizer.")
        rows.append(peak)
    return pd.DataFrame(rows)


def _alternative_peaks() -> pd.DataFrame:
    full_path = ROBUSTNESS_DIR / "alternative_ppml_product_promotion" / "alternative_ppml_full_grid.csv"
    full = pd.read_csv(full_path)
    peaks = _assert_peak(full, "alternative_product_promotion")
    rows = []
    for peak in peaks.itertuples(index=False):
        denominator = float(peak.myopic_profit)
        if not np.isfinite(denominator) or denominator <= 0:
            raise AssertionError("Alternative demand model has an invalid 12-week myopic denominator.")
        rows.append({
            "specification": "alternative_product_promotion",
            "capacity": int(peak.capacity),
            "peak_lambda": float(peak.reimbursement_share),
            "peak_delta_total": float(peak.delta_total),
            "myopic_profit_48w": denominator,
            "peak_delta_total_pct_48w": 100.0 * float(peak.delta_total) / denominator,
        })
    return pd.DataFrame(rows)


def _cooldown_peaks() -> pd.DataFrame:
    path = ROBUSTNESS_DIR / "cooldown_3_weeks" / "cooldown_3_weeks_full_grid.csv"
    full = pd.read_csv(path)
    peaks = _assert_peak(full, "cooldown_3_weeks")
    required = {"myopic_profit"}
    missing = required.difference(peaks.columns)
    if missing:
        raise ValueError(f"Cooldown-3 grid lacks {sorted(missing)}")
    rows = []
    for peak in peaks.itertuples(index=False):
        denominator = float(peak.myopic_profit)
        percent = 100.0 * float(peak.delta_total) / denominator
        rows.append({"specification": "cooldown_3_weeks", "capacity": int(peak.capacity),
                     "peak_lambda": float(peak.reimbursement_share),
                     "peak_delta_total": float(peak.delta_total),
                     "myopic_profit_48w": denominator,
                     "peak_delta_total_pct_48w": percent})
    return pd.DataFrame(rows)


def build_robustness_peak_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long and compact peak table, validating each cell's denominator."""
    main = load_pickle(MAIN_ARTIFACT)["policy_results"]
    task7 = load_pickle(TASK7_ARTIFACT)
    full_grids = {
        "main": main,
        "pooled_displacement": task7["pooled_displacement"]["results"],
        "exclude_3800001611": task7["exclude_3800001611"]["results"],
    }
    rows = []
    for specification, full_grid in full_grids.items():
        peaks = _assert_peak(full_grid, specification)
        for peak in peaks.itertuples(index=False):
            denominator = float(peak.myopic_profit)
            if not np.isfinite(denominator) or denominator <= 0:
                raise AssertionError(f"{specification}, B={peak.capacity}: invalid own 12-week denominator.")
            percent = 100.0 * float(peak.delta_total) / denominator
            rows.append({
                "specification": specification,
                "capacity": int(peak.capacity),
                "peak_lambda": float(peak.reimbursement_share),
                "peak_delta_total": float(peak.delta_total),
                "myopic_profit_48w": denominator,
                "peak_delta_total_pct_48w": percent,
            })
    long = pd.concat([pd.DataFrame(rows), _cooldown_peaks(), _alternative_peaks()], ignore_index=True)
    if long.groupby(["specification", "capacity"], observed=True).size().ne(1).any():
        raise AssertionError("Robustness peak table does not have exactly one cell per specification/capacity.")
    long["cell"] = long.apply(
        lambda row: f"{row.peak_lambda:.2f} ({row.peak_delta_total_pct_48w:.2f}%)", axis=1
    )
    order = list(LABELS)
    compact = (
        long.assign(specification=lambda x: pd.Categorical(x.specification, order, ordered=True))
        .pivot(index="specification", columns="capacity", values="cell")
        .reindex(order)[list(CAPACITIES)]
        .rename(index=LABELS, columns=lambda b: f"B={b}")
        .reset_index(names="Specification")
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(TABLE_DIR / "robustness_peak_location_and_magnitude_long.csv", index=False)
    compact.to_csv(TABLE_DIR / "robustness_peak_location_and_magnitude.csv", index=False)
    return long, compact


if __name__ == "__main__":
    _, table = build_robustness_peak_table()
    print(table.to_string(index=False))
