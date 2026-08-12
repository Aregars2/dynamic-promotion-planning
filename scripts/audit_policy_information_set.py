"""Audit the persisted policy forecast artifact for future-row dependencies.

This is diagnostic only: it does not refit PPML or regenerate policy results.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dynamic_promotion_planning.policy import load_pickle


def main() -> None:
    tables = ROOT / "results" / "final" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    policy = load_pickle(ROOT / "artifacts" / "policy" / "policy_optimization.pkl")
    predictions = load_pickle(ROOT / "artifacts" / "demand" / "policy_common_origin_predictions.pkl")
    origin = int(policy["source_weeks"][0])
    weeks = list(range(origin, origin + 12))
    subset = predictions.loc[predictions["week"].isin(weeks)].copy()
    if subset.empty:
        raise ValueError("Persisted policy predictions do not cover the 12-week horizon.")
    classifications = {
        "planning_origin_week": "known at planning origin",
        "forecast_week": "deterministic future-calendar information",
        "fit_last_week": "known at planning origin",
        "model_unit_price": "revealed only after origin (realized future row)",
        "regular_price": "revealed only after origin (realized future row)",
        "promotion_indicator": "revealed only after origin (realized future row)",
        "post_promotion_indicator": "revealed only after origin (realized future row)",
        "discount_depth_model": "revealed only after origin (realized future row)",
        "price_imputed": "revealed only after origin (realized future row)",
        "reference_price": "revealed only after origin (realized future row)",
        "calendar_month": "deterministic future-calendar information",
        "scaled_time_trend": "deterministic future-calendar information",
        "holiday indicators": "deterministic future-calendar information",
        "lagged outcomes": "not in policy formula; present in rolling context only",
    }
    audit = pd.DataFrame(
        [{"predictor": predictor, "classification": classification}
         for predictor, classification in classifications.items()]
    )
    audit["planning_origin_week"] = origin
    audit.to_csv(tables / "policy_information_set_predictor_audit.csv", index=False)
    coverage = subset.groupby("week", observed=True).agg(
        prediction_rows=("store_upc", "size"),
        realized_panel_rows=("store_upc", "nunique"),
        fit_last_week_min=("fit_last_week", "min"),
        fit_last_week_max=("fit_last_week", "max"),
    ).reset_index().rename(columns={"week": "forecast_week"})
    coverage["planning_origin_week"] = origin
    coverage["complete_fixed_grid_verified"] = False
    coverage.to_csv(tables / "policy_information_set_grid_audit.csv", index=False)
    print(f"Current artifact audit: origin={origin}; rows={len(subset):,}; forecast weeks={len(coverage)}")
    print("Classification: ex_post_conditional_path (future realized covariates and realized row availability are used).")


if __name__ == "__main__":
    main()
