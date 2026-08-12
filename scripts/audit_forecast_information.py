"""Audit whether policy demand profiles are common-origin ex-ante forecasts."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dynamic_promotion_planning.forecast_audit import audit_forecast_information
from dynamic_promotion_planning.policy import build_weekly_economic_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--future-covariates-verified",
        action="store_true",
        help=(
            "Declare that every covariate used for all profile weeks was available "
            "at the common planning origin. Do not set this without a separate audit."
        ),
    )
    args = parser.parse_args()

    prediction_path = (
        PROJECT_ROOT / "artifacts" / "demand" / "demand_predictions.pkl"
    )
    policy_path = (
        PROJECT_ROOT / "artifacts" / "policy" / "policy_optimization.pkl"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)

    predictions = pd.read_pickle(prediction_path)
    artifact = pd.read_pickle(policy_path)
    selected_weeks = artifact["source_weeks"]

    audit_rows = predictions.copy()
    if "split" in audit_rows.columns:
        audit_rows = audit_rows.loc[
            audit_rows["split"].astype(str).str.lower().eq("test")
        ]
    if "model" in audit_rows.columns:
        audit_rows = audit_rows.loc[
            audit_rows["model"].astype(str).eq("product_promotion")
        ]

    audit = audit_forecast_information(
        audit_rows,
        selected_weeks=selected_weeks,
        planning_origin_week=min(selected_weeks),
        future_covariates_verified=args.future_covariates_verified,
    )
    print(pd.Series(audit.to_dict()).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
