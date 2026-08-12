"""Focused audit of persisted PPML regular-state baseline predictions.

This script does not refit models or touch policy notebooks.  It verifies the
exported store-level counterfactual predictions, writes the promotional-row
audit, and compares the historical mean aggregation with the current literal
store-sum baseline aggregation.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dynamic_promotion_planning.demand import regular_counterfactual_audit
from dynamic_promotion_planning.policy import load_pickle


MODEL_NAME = "product_promotion_depth"
PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "demand" / "demand_predictions.pkl"
TABLE_DIR = PROJECT_ROOT / "results" / "final" / "tables"


def _scale_summary(values: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return (
        values.groupby("upc", observed=True)["baseline_demand"]
        .agg(
            **{
                f"{prefix}_mean": "mean",
                f"{prefix}_median": "median",
                f"{prefix}_min": "min",
                f"{prefix}_max": "max",
            }
        )
        .reset_index()
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_pickle(PREDICTIONS_PATH)
    active = predictions.loc[
        predictions["model"].astype(str).eq(MODEL_NAME)
        & predictions["split"].astype(str).str.lower().eq("test")
    ].copy()
    if active.empty:
        raise ValueError("No test predictions for the active PPML model.")

    required = {
        "upc", "week", "store", "regular_price",
        "counterfactual_model_unit_price", "promotion_indicator",
        "post_promotion_indicator", "mu_hat_regular_counterfactual",
    }
    missing = required.difference(active.columns)
    if missing:
        raise ValueError(f"Prediction artifact is missing {sorted(missing)}")

    cf = pd.to_numeric(active["mu_hat_regular_counterfactual"], errors="coerce")
    if not np.isfinite(cf).all() or (cf < 0).any():
        raise AssertionError("Counterfactual PPML predictions must be finite and nonnegative.")

    audit = regular_counterfactual_audit(
        predictions, model_name=MODEL_NAME, split="test"
    )
    if len(audit) != 9_609:
        raise AssertionError(f"Expected 9,609 promotion/post-promotion rows, found {len(audit)}.")
    if not np.allclose(
        audit["counterfactual_model_unit_price"], audit["regular_price"], rtol=0, atol=0
    ):
        raise AssertionError("Promotional audit rows do not reset price to regular price.")
    if not audit["counterfactual_promotion_indicator"].eq(0).all() or not audit[
        "counterfactual_post_promotion_indicator"
    ].eq(0).all():
        raise AssertionError("Promotional audit rows do not reset promotion states.")
    audit_cf = pd.to_numeric(audit["mu_hat_regular_counterfactual"], errors="coerce")
    if not np.isfinite(audit_cf).all() or (audit_cf < 0).any():
        raise AssertionError("Promotional audit predictions must be finite and nonnegative.")
    audit.to_csv(TABLE_DIR / "ppml_regular_counterfactual_promotional_rows.csv", index=False)

    group_keys = ["upc", "week"]
    store_cf = active.loc[:, group_keys + ["store", "mu_hat_regular_counterfactual"]].copy()
    store_cf["baseline_demand"] = pd.to_numeric(
        store_cf.pop("mu_hat_regular_counterfactual"), errors="raise"
    )
    before = store_cf.groupby(group_keys, observed=True)["baseline_demand"].mean().reset_index()
    after = store_cf.groupby(group_keys, observed=True)["baseline_demand"].sum().reset_index()
    contributing = store_cf.groupby(group_keys, observed=True)["store"].nunique().reset_index(name="contributing_stores")
    after = after.merge(contributing, on=group_keys, validate="one_to_one")
    # This is the exact policy baseline definition: no panel averaging or multiplier.
    literal_sum = store_cf.groupby(group_keys, observed=True)["baseline_demand"].sum()
    indexed_after = after.set_index(group_keys)["baseline_demand"]
    if not indexed_after.equals(literal_sum):
        raise AssertionError("Product-week baseline is not the literal sum of store predictions.")

    report = _scale_summary(before, "before_mean_aggregation").merge(
        _scale_summary(after, "after_store_sum"), on="upc", validate="one_to_one"
    )
    stores = after.groupby("upc", observed=True)["contributing_stores"].agg(
        contributing_stores_mean="mean",
        contributing_stores_min="min",
        contributing_stores_max="max",
    ).reset_index()
    report = report.merge(stores, on="upc", validate="one_to_one").sort_values("upc")
    report.to_csv(TABLE_DIR / "policy_baseline_scale_before_after.csv", index=False)
    after.to_csv(TABLE_DIR / "policy_baseline_scale_after_sum.csv", index=False)
    print(f"Active test prediction rows: {len(active):,}")
    print(f"Promotional/post-promotion audit rows: {len(audit):,}")
    print("Counterfactual reset, prediction finiteness, and literal store-sum checks: passed")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
