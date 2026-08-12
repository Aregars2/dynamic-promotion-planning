"""Audits for information availability in weekly policy profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastInformationAudit:
    """Summary of whether a profile uses one common pre-decision model fit."""

    planning_origin_week: int
    first_profile_week: int
    last_profile_week: int
    profile_week_count: int
    row_count: int
    fit_metadata_available: bool
    maximum_fit_week: int | None
    within_horizon_refit_detected: bool | None
    common_origin_fit_verified: bool
    future_covariates_verified: bool
    classification: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_forecast_information(
    predictions: pd.DataFrame,
    selected_weeks: Sequence[int],
    *,
    planning_origin_week: int | None = None,
    week_column: str = "week",
    fit_last_week_column: str = "fit_last_week",
    future_covariates_verified: bool = False,
) -> ForecastInformationAudit:
    """Classify the model-fit information set used by a policy profile.

    This audit verifies only model-estimation timing. The caller must separately
    establish that every prediction covariate was available at the planning origin.
    """
    weeks = sorted({int(value) for value in selected_weeks})
    if not weeks:
        raise ValueError("selected_weeks must contain at least one week.")
    origin = int(planning_origin_week if planning_origin_week is not None else weeks[0])

    if week_column not in predictions.columns:
        raise ValueError(f"Prediction data are missing {week_column!r}.")

    frame = predictions.copy()
    frame[week_column] = pd.to_numeric(frame[week_column], errors="coerce")
    frame = frame.loc[frame[week_column].isin(weeks)].copy()
    if frame.empty:
        raise ValueError("No prediction rows match the selected profile weeks.")

    if fit_last_week_column not in frame.columns:
        return ForecastInformationAudit(
            planning_origin_week=origin,
            first_profile_week=weeks[0],
            last_profile_week=weeks[-1],
            profile_week_count=len(weeks),
            row_count=len(frame),
            fit_metadata_available=False,
            maximum_fit_week=None,
            within_horizon_refit_detected=None,
            common_origin_fit_verified=False,
            future_covariates_verified=bool(future_covariates_verified),
            classification="unverifiable_missing_fit_metadata",
        )

    fit_last_week = pd.to_numeric(
        frame[fit_last_week_column],
        errors="coerce",
    )
    if fit_last_week.isna().any():
        return ForecastInformationAudit(
            planning_origin_week=origin,
            first_profile_week=weeks[0],
            last_profile_week=weeks[-1],
            profile_week_count=len(weeks),
            row_count=len(frame),
            fit_metadata_available=False,
            maximum_fit_week=None,
            within_horizon_refit_detected=None,
            common_origin_fit_verified=False,
            future_covariates_verified=bool(future_covariates_verified),
            classification="unverifiable_incomplete_fit_metadata",
        )

    maximum_fit_week = int(fit_last_week.max())
    within_horizon_refit = bool((fit_last_week >= origin).any())
    common_origin_fit = not within_horizon_refit

    if not common_origin_fit:
        classification = "rolling_conditional_path"
    elif future_covariates_verified:
        classification = "ex_ante_common_origin"
    else:
        classification = "common_origin_fit_covariates_unverified"

    return ForecastInformationAudit(
        planning_origin_week=origin,
        first_profile_week=weeks[0],
        last_profile_week=weeks[-1],
        profile_week_count=len(weeks),
        row_count=len(frame),
        fit_metadata_available=True,
        maximum_fit_week=maximum_fit_week,
        within_horizon_refit_detected=within_horizon_refit,
        common_origin_fit_verified=common_origin_fit,
        future_covariates_verified=bool(future_covariates_verified),
        classification=classification,
    )


def assert_ex_ante_common_origin(
    audit: ForecastInformationAudit,
) -> None:
    """Raise unless model timing and future-covariate availability are verified."""
    if audit.classification != "ex_ante_common_origin":
        raise AssertionError(
            "Weekly policy profile is not verified as an ex-ante common-origin "
            f"forecast. Classification: {audit.classification}."
        )


def assert_future_outcome_invariance(
    original_profile: pd.DataFrame,
    perturbed_profile: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    key_columns: Sequence[str] = ("upc", "week"),
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> None:
    """Assert that future-outcome perturbation leaves forecast values unchanged."""
    columns = [*key_columns, *value_columns]
    missing_original = sorted(set(columns).difference(original_profile.columns))
    missing_perturbed = sorted(set(columns).difference(perturbed_profile.columns))
    if missing_original or missing_perturbed:
        raise ValueError(
            "Profile comparison columns are missing: "
            f"original={missing_original}, perturbed={missing_perturbed}."
        )

    left = (
        original_profile.loc[:, columns]
        .sort_values(list(key_columns))
        .reset_index(drop=True)
    )
    right = (
        perturbed_profile.loc[:, columns]
        .sort_values(list(key_columns))
        .reset_index(drop=True)
    )
    if not left.loc[:, key_columns].equals(right.loc[:, key_columns]):
        raise AssertionError("Profile keys changed after future-outcome perturbation.")

    for column in value_columns:
        np.testing.assert_allclose(
            pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float),
            pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float),
            atol=atol,
            rtol=rtol,
        )
