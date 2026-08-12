"""Fit only the policy PPML model and export its ex-ante fixed-grid forecast."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_promotion_planning.demand import (
    POLICY_COMMON_CONTROLS,
    common_origin_ex_ante_predictions,
)


def run() -> Path:
    """Create the sole demand artifact required by Notebook 06."""
    selected = pd.read_parquet(ROOT / "data" / "processed" / "paper_selected_sample.parquet")
    panel_ids = selected["store_upc"].astype(str).unique()
    products = selected["upc"].astype(str).unique()
    source = pd.read_parquet(ROOT / "data" / "processed" / "cereal_demand_model_data.parquet")
    source["store_upc"] = source["store_upc"].astype(str)
    source["upc"] = source["upc"].astype(str)
    frame = source.loc[source["store_upc"].isin(panel_ids) & source["upc"].isin(products)].copy()
    frame = frame.sort_values(["store_upc", "week"], kind="mergesort").reset_index(drop=True)
    state = frame["pricing_state"].astype("string").str.strip().str.lower()
    frame["promotion_indicator"] = state.eq("promotion").astype("int8")
    frame["post_promotion_indicator"] = state.eq("post_promotion").astype("int8")
    frame["log_price_model"] = np.log(pd.to_numeric(frame["model_unit_price"], errors="coerce"))
    frame["discount_depth_model"] = pd.to_numeric(frame["discount_depth"], errors="coerce").fillna(0.0)
    frame["discount_depth_sq"] = frame["discount_depth_model"] ** 2
    frame["price_imputed_indicator"] = frame["price_imputed"].fillna(False).astype(int)
    for column in ("store_upc", "upc", "calendar_month"):
        frame[column] = frame[column].astype("category")
    required = [
        "move", "log_price_model", "promotion_indicator", "post_promotion_indicator",
        "discount_depth_model", "discount_depth_sq", "calendar_month", "scaled_time_trend",
        "thanksgiving_week", "christmas_week", "new_year_week", "easter_week",
        "price_imputed_indicator", "store_upc", "upc", "week", "regular_price",
    ]
    frame = frame.dropna(subset=required).copy()
    weeks = np.sort(frame["week"].unique())
    origin_index = int(np.floor(0.80 * len(weeks)))
    forecast_weeks = weeks[origin_index:]
    origin = int(forecast_weeks.min())
    predictions = common_origin_ex_ante_predictions(
        frame, forecast_weeks, "product_promotion_depth",
        planning_origin_week=origin, common_controls=POLICY_COMMON_CONTROLS,
    )
    predictions["split"] = "test"
    target = ROOT / "artifacts" / "demand" / "policy_common_origin_predictions.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_pickle(target)
    assert predictions["prediction_design"].eq("ex_ante_fixed_grid").all()
    assert predictions["origin_predictors_verified"].all()
    print(f"Saved {len(predictions):,} ex-ante rows to {target}")
    return target


if __name__ == "__main__":
    run()
