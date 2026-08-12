from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_promotion_planning.forecast_audit import (
    assert_ex_ante_common_origin,
    assert_future_outcome_invariance,
    audit_forecast_information,
)
from dynamic_promotion_planning.demand import (
    POLICY_COMMON_CONTROLS,
    common_origin_predictions,
)


def test_common_origin_forecast_passes_when_fit_precedes_origin() -> None:
    predictions = pd.DataFrame(
        {
            "week": [100, 101, 102, 103],
            "fit_last_week": [99, 99, 99, 99],
            "mu_hat": [10.0, 11.0, 12.0, 13.0],
        }
    )
    audit = audit_forecast_information(
        predictions,
        selected_weeks=[100, 101, 102, 103],
        planning_origin_week=100,
        future_covariates_verified=True,
    )
    assert audit.classification == "ex_ante_common_origin"
    assert audit.common_origin_fit_verified
    assert not audit.within_horizon_refit_detected
    assert_ex_ante_common_origin(audit)


def test_rolling_refits_inside_horizon_are_classified_as_conditional_path() -> None:
    predictions = pd.DataFrame(
        {
            "week": [100, 101, 102, 103, 104, 105],
            "fit_last_week": [99, 99, 99, 99, 103, 103],
            "mu_hat": [10.0] * 6,
        }
    )
    audit = audit_forecast_information(
        predictions,
        selected_weeks=range(100, 106),
        planning_origin_week=100,
        future_covariates_verified=True,
    )
    assert audit.classification == "rolling_conditional_path"
    assert audit.within_horizon_refit_detected
    assert not audit.common_origin_fit_verified
    with pytest.raises(AssertionError, match="rolling_conditional_path"):
        assert_ex_ante_common_origin(audit)


def test_common_fit_without_covariate_audit_is_not_declared_ex_ante() -> None:
    predictions = pd.DataFrame(
        {
            "week": [100, 101],
            "fit_last_week": [99, 99],
        }
    )
    audit = audit_forecast_information(
        predictions,
        selected_weeks=[100, 101],
        planning_origin_week=100,
        future_covariates_verified=False,
    )
    assert audit.classification == "common_origin_fit_covariates_unverified"
    with pytest.raises(AssertionError):
        assert_ex_ante_common_origin(audit)


def test_missing_fit_metadata_is_reported_as_unverifiable() -> None:
    predictions = pd.DataFrame({"week": [100, 101], "mu_hat": [1.0, 1.0]})
    audit = audit_forecast_information(
        predictions,
        selected_weeks=[100, 101],
    )
    assert audit.classification == "unverifiable_missing_fit_metadata"
    assert not audit.fit_metadata_available


def test_policy_controls_exclude_realized_sales_lags() -> None:
    assert "log1p_lag_move" not in POLICY_COMMON_CONTROLS


def test_common_origin_requires_the_first_forecast_week() -> None:
    with pytest.raises(ValueError, match="first forecast week"):
        common_origin_predictions(
            pd.DataFrame(),
            np.array([101, 102]),
            "product_promotion",
            planning_origin_week=100,
        )


def test_future_outcome_invariance_check_accepts_identical_profiles() -> None:
    original = pd.DataFrame(
        {
            "upc": ["A", "A"],
            "week": [100, 101],
            "demand_factor": [0.9, 1.1],
        }
    )
    perturbed = original.iloc[::-1].reset_index(drop=True)
    assert_future_outcome_invariance(
        original,
        perturbed,
        value_columns=["demand_factor"],
    )


def test_future_outcome_invariance_check_detects_profile_change() -> None:
    original = pd.DataFrame(
        {
            "upc": ["A", "A"],
            "week": [100, 101],
            "demand_factor": [0.9, 1.1],
        }
    )
    perturbed = original.copy()
    perturbed.loc[1, "demand_factor"] = 2.0
    with pytest.raises(AssertionError):
        assert_future_outcome_invariance(
            original,
            perturbed,
            value_columns=["demand_factor"],
        )
