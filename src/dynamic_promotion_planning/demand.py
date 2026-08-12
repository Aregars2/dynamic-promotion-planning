"""Demand-sample selection, PPML specification, and rolling prediction helpers."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize
from scipy.special import logsumexp

from .config import DemandConfig


DEFAULT_COMMON_CONTROLS = (
    " + log1p_lag_move"
    " + log1p_lag_move_mean_4"
    " + price_imputed_indicator"
    " + C(calendar_month)"
    " + scaled_time_trend"
    " + thanksgiving_week"
    " + christmas_week"
    " + new_year_week"
    " + easter_week"
)

# Policy forecasts may use only variables known at the planning origin.
POLICY_COMMON_CONTROLS = (
    " + price_imputed_indicator"
    " + C(calendar_month)"
    " + scaled_time_trend"
    " + thanksgiving_week"
    " + christmas_week"
    " + new_year_week"
    " + easter_week"
)

ROLLING_OUTPUT_COLUMNS = [
    "store",
    "upc",
    "store_upc",
    "week",
    "week_start",
    "move",
    "model_unit_price",
    "unit_price_observed",
    "regular_price",
    "discount_depth_model",
    "price_imputed",
    "pricing_state",
    "promotion_indicator",
    "post_promotion_indicator",
    "reference_price",
]


def period_availability(
    frame: pd.DataFrame,
    group_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Calculate panel availability without using future sales totals."""
    return frame.groupby(group_column, observed=True).agg(
        **{
            f"{prefix}_rows": ("move", "size"),
            f"{prefix}_weeks": ("week", "nunique"),
            f"{prefix}_stores": ("store", "nunique"),
        }
    )


def _validate_product_selection_columns(frame: pd.DataFrame) -> None:
    required = {
        "upc",
        "store",
        "week",
        "move",
        "model_unit_price",
        "price_imputed",
        "promotion_indicator",
        "regular_indicator",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Product-selection data are missing columns: " f"{sorted(missing)}"
        )


def _training_product_summary(train_data: pd.DataFrame) -> pd.DataFrame:
    return train_data.groupby("upc", observed=True).agg(
        train_units=("move", "sum"),
        train_rows=("move", "size"),
        train_weeks=("week", "nunique"),
        train_stores=("store", "nunique"),
        train_promotion_rows=("promotion_indicator", "sum"),
        train_regular_rows=("regular_indicator", "sum"),
        train_imputed_share=("price_imputed", "mean"),
    )


def _observed_price_summary(train_data: pd.DataFrame) -> pd.DataFrame:
    observed = train_data.loc[~train_data["price_imputed"]]
    return observed.groupby("upc", observed=True).agg(
        train_distinct_prices=("model_unit_price", "nunique"),
        train_log_price_sd=("model_unit_price", lambda values: np.log(values).std()),
        train_price_p10=("model_unit_price", lambda values: values.quantile(0.10)),
        train_price_p90=("model_unit_price", lambda values: values.quantile(0.90)),
    )


def _selection_criteria(
    selection: pd.DataFrame,
    config: DemandConfig,
) -> dict[str, pd.Series]:
    return {
        "adequate_train_weeks": selection["train_weeks"].ge(
            config.min_product_train_weeks
        ),
        "adequate_calibration_weeks": selection["calibration_weeks"].ge(
            config.min_product_calibration_weeks
        ),
        "adequate_test_weeks": selection["test_weeks"].ge(
            config.min_product_test_weeks
        ),
        "adequate_store_coverage": selection["train_stores"].ge(
            config.min_product_train_stores
        ),
        "adequate_distinct_prices": selection["train_distinct_prices"].ge(
            config.min_product_distinct_prices
        ),
        "adequate_promotions": selection["train_promotion_rows"].ge(
            config.min_product_promotion_rows
        ),
        "adequate_regular_rows": selection["train_regular_rows"].ge(
            config.min_product_regular_rows
        ),
        "acceptable_imputation": selection["train_imputed_share"].le(
            config.max_product_imputed_share
        ),
        "adequate_price_range": selection["train_relative_price_range"].ge(
            config.min_product_relative_price_range
        ),
    }


def build_product_selection_table(
    frame: pd.DataFrame,
    train_week_values: np.ndarray,
    calibration_week_values: np.ndarray,
    test_week_values: np.ndarray,
    config: DemandConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Construct the chronological product-eligibility audit table."""
    config = config or DemandConfig()
    _validate_product_selection_columns(frame)

    train_data = frame.loc[frame["week"].isin(train_week_values)]
    calibration_data = frame.loc[frame["week"].isin(calibration_week_values)]
    test_data = frame.loc[frame["week"].isin(test_week_values)]

    selection = (
        _training_product_summary(train_data)
        .join(_observed_price_summary(train_data), how="left")
        .join(period_availability(calibration_data, "upc", "calibration"), how="left")
        .join(period_availability(test_data, "upc", "test"), how="left")
        .fillna(0)
        .reset_index()
    )
    selection["train_relative_price_range"] = (
        selection["train_price_p90"]
        / selection["train_price_p10"].replace(0, np.nan)
        - 1.0
    )

    criteria = _selection_criteria(selection, config)
    for name, criterion in criteria.items():
        selection[name] = criterion
    criterion_columns = list(criteria)
    selection["eligible"] = selection[criterion_columns].all(axis=1)
    selection = selection.sort_values(
        ["eligible", "train_units"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return selection, criterion_columns


def panel_period_counts(
    frame: pd.DataFrame,
    week_values: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    """Count panel rows and distinct weeks in a specified temporal partition."""
    return (
        frame.loc[frame["week"].isin(week_values)]
        .groupby("store_upc", observed=True)
        .agg(
            **{
                f"{prefix}_weeks": ("week", "nunique"),
                f"{prefix}_rows": ("move", "size"),
            }
        )
    )


def poisson_deviance_contribution(
    observed: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return observation-level Poisson deviance contributions."""
    y = np.asarray(observed, dtype=float)
    mu = np.clip(np.asarray(predicted, dtype=float), 1e-12, None)
    log_term = np.zeros_like(y, dtype=float)
    positive = y > 0
    log_term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    return 2.0 * (log_term - (y - mu))


def prediction_metrics(frame: pd.DataFrame) -> pd.Series:
    """Summarize predictive calibration and error for a prediction frame."""
    y = frame["move"].to_numpy(dtype=float)
    mu = frame["mu_hat"].to_numpy(dtype=float)
    deviance = poisson_deviance_contribution(y, mu)
    return pd.Series(
        {
            "n": len(frame),
            "mean_observed": y.mean(),
            "mean_predicted": mu.mean(),
            "observed_to_predicted_ratio": y.sum() / mu.sum(),
            "mae": np.mean(np.abs(y - mu)),
            "rmse": np.sqrt(np.mean((y - mu) ** 2)),
            "mean_poisson_deviance": deviance.mean(),
            "median_poisson_deviance": np.median(deviance),
            "p90_poisson_deviance": np.quantile(deviance, 0.90),
        }
    )


def paired_weekly_loss_test(
    frame: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    label: str,
    max_lags: int = 4,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Compare two models' paired weekly mean Poisson-deviance losses."""
    import statsmodels.api as sm

    weekly = (
        frame.loc[frame["model"].isin([model_a, model_b]), ["week", "model", "poisson_deviance"]]
        .pivot_table(index="week", columns="model", values="poisson_deviance", aggfunc="mean")
        .dropna().sort_index()
    )
    difference = weekly[model_a] - weekly[model_b]
    if len(difference) < 3:
        raise ValueError(f"{label}: fewer than three paired weeks.")
    fitted = sm.OLS(difference.to_numpy(), np.ones((len(difference), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(max_lags, len(difference) - 1)}
    )
    ci_lower, ci_upper = fitted.conf_int()[0]
    return pd.Series({
        "comparison": label, "weeks": len(difference),
        "mean_deviance_difference": difference.mean(), "hac_se": fitted.bse[0],
        "t_statistic": fitted.tvalues[0], "p_value": fitted.pvalues[0],
        "ci_lower": ci_lower, "ci_upper": ci_upper,
    }), weekly, difference


def spline_expression(lower_bound: float, upper_bound: float) -> str:
    """Return the Patsy expression for the fixed four-degree-of-freedom spline."""
    return (
        "bs("
        "log_price_spline, "
        "df=4, "
        "degree=3, "
        "include_intercept=False, "
        f"lower_bound={float(lower_bound)!r}, "
        f"upper_bound={float(upper_bound)!r}"
        ")"
    )


def build_formula(
    model_name: str,
    lower_bound: float,
    upper_bound: float,
    common_controls: str = DEFAULT_COMMON_CONTROLS,
) -> str:
    """Construct one of the pre-specified PPML demand formulas."""
    promotion_terms = {
        "common_spline": (
            " + promotion_indicator" " + post_promotion_indicator"
        ),
        "product_promotion": (
            " + C(upc):promotion_indicator"
            " + C(upc):post_promotion_indicator"
        ),
        "product_promotion_depth": (
            " + C(upc):promotion_indicator"
            " + C(upc):post_promotion_indicator"
            " + discount_depth_model"
            " + discount_depth_sq"
        ),
    }
    if model_name not in promotion_terms:
        raise ValueError(f"Unknown model: {model_name}")
    return (
        "move ~ 0 + C(store_upc) + "
        + spline_expression(lower_bound, upper_bound)
        + promotion_terms[model_name]
        + common_controls
    )


def prepare_price_for_spline(
    fit_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Estimate training-only spline bounds and clip both data partitions."""
    fit_copy = fit_data.copy()
    prediction_copy = prediction_data.copy()
    lower = float(fit_copy["log_price_model"].quantile(lower_quantile))
    upper = float(fit_copy["log_price_model"].quantile(upper_quantile))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Spline bounds are not finite.")
    if lower >= upper:
        raise ValueError("Spline bounds are invalid.")
    fit_copy["log_price_spline"] = fit_copy["log_price_model"].clip(lower, upper)
    prediction_copy["log_price_spline"] = prediction_copy[
        "log_price_model"
    ].clip(lower, upper)
    return fit_copy, prediction_copy, lower, upper


def _forecast_blocks(
    forecast_week_values: np.ndarray,
    refit_every_n_weeks: int,
) -> Iterator[tuple[int, np.ndarray]]:
    weeks = np.asarray(sorted(forecast_week_values))
    for block_number, block_start in enumerate(
        range(0, len(weeks), refit_every_n_weeks)
    ):
        yield block_number, weeks[block_start : block_start + refit_every_n_weeks]


def _rolling_fit_and_prediction_data(
    frame: pd.DataFrame,
    block_weeks: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    first_forecast_week = block_weeks[0]
    fit_data = frame.loc[frame["week"] < first_forecast_week].copy()
    prediction_data = frame.loc[frame["week"].isin(block_weeks)].copy()
    if fit_data.empty or prediction_data.empty:
        return None
    return fit_data, prediction_data


def _validate_ppml_design(fit_data: pd.DataFrame, model_name: str) -> None:
    if not np.isfinite(fit_data["move"].to_numpy(float)).all():
        raise ValueError(f"{model_name}: nonfinite outcome values.")
    zero_sales_panels = (
        fit_data.groupby("store_upc", observed=True)["move"].sum().eq(0)
    )
    if zero_sales_panels.any():
        raise ValueError(
            f"{model_name}: {zero_sales_panels.sum()} store-product panels "
            "have zero total sales in the estimation window."
        )


@dataclass
class _ProfiledPoissonResult:
    params: np.ndarray
    design_info: patsy.DesignInfo
    panel_scales: pd.Series
    converged: bool = True

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        design = patsy.build_design_matrices([self.design_info], data)[0]
        scales = data["store_upc"].map(self.panel_scales)
        if scales.isna().any():
            raise ValueError("Prediction panel lacks a recovered fixed effect.")
        return scales.to_numpy(float) * np.exp(
            np.clip(np.asarray(design) @ self.params, -30.0, 30.0)
        )


def _fit_ppml(
    fit_data: pd.DataFrame,
    formula: str,
    model_name: str,
    block_weeks: np.ndarray,
):
    _validate_ppml_design(fit_data, model_name)
    rhs = formula.replace("0 + C(store_upc) + ", "0 + ", 1).split("~", 1)[1]
    design = patsy.dmatrix(rhs, fit_data, return_type="dataframe")
    x = design.to_numpy(float)
    y = fit_data["move"].to_numpy(float)
    codes, panels = pd.factorize(fit_data["store_upc"], sort=True)
    totals = np.bincount(codes, weights=y)
    if (totals <= 0).any():
        raise ValueError(f"{model_name}: zero-sales panel in estimation window.")
    def objective(beta):
        eta = x @ beta
        denominator = np.array([logsumexp(eta[codes == g]) for g in range(len(panels))])
        mu = totals[codes] * np.exp(eta - denominator[codes])
        return float(totals @ denominator - y @ eta), x.T @ (mu - y)
    try:
        result = minimize(
            objective,
            np.zeros(x.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1_000, "gtol": 1e-7, "ftol": 1e-12},
        )
    except (ValueError, FloatingPointError) as error:
        raise RuntimeError(
            f"PPML failed for {model_name}; forecast weeks "
            f"{int(block_weeks.min())}–{int(block_weeks.max())}; training weeks "
            f"{int(fit_data['week'].min())}–{int(fit_data['week'].max())}."
        ) from error
    gradient_maximum = float(np.abs(result.jac).max())
    if not result.success and gradient_maximum > 1e-5:
        raise RuntimeError(
            f"PPML did not converge for {model_name}; forecast weeks "
            f"{int(block_weeks.min())}–{int(block_weeks.max())}: {result.message}; "
            f"maximum score={gradient_maximum:.3e}"
        )
    eta = x @ result.x
    denominator = np.array([logsumexp(eta[codes == g]) for g in range(len(panels))])
    return _ProfiledPoissonResult(result.x, design.design_info, pd.Series(totals / np.exp(denominator), index=panels))


def _result_converged(result) -> bool:
    """Return optimizer convergence for either IRLS or likelihood results."""
    if hasattr(result, "converged"):
        return bool(result.converged)
    return bool(getattr(result, "mle_retvals", {}).get("converged", False))


def _fit_rolling_block(
    frame: pd.DataFrame,
    block_weeks: np.ndarray,
    model_name: str,
    common_controls: str,
):
    partitions = _rolling_fit_and_prediction_data(frame, block_weeks)
    if partitions is None:
        return None
    fit_data, prediction_data = partitions
    fit_data, prediction_data, lower, upper = prepare_price_for_spline(
        fit_data,
        prediction_data,
    )
    formula = build_formula(
        model_name,
        lower_bound=lower,
        upper_bound=upper,
        common_controls=common_controls,
    )
    result = _fit_ppml(
        fit_data,
        formula,
        model_name,
        block_weeks,
    )
    return result, fit_data, prediction_data, lower, upper


def _prediction_block(
    result,
    prediction_data: pd.DataFrame,
    *,
    model_name: str,
    block_number: int,
    fit_last_week: int,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Score observed and regular-state counterfactual forecast inputs."""
    counterfactual = prediction_data.copy()
    regular_price = pd.to_numeric(
        counterfactual["regular_price"],
        errors="coerce",
    )
    if regular_price.isna().any() or regular_price.le(0).any():
        raise ValueError("Counterfactual regular prices must be finite and positive.")
    counterfactual["model_unit_price"] = regular_price
    counterfactual["log_price_model"] = np.log(regular_price)
    counterfactual["log_price_spline"] = counterfactual[
        "log_price_model"
    ].clip(lower, upper)
    for column in ["promotion_indicator", "post_promotion_indicator"]:
        counterfactual[column] = 0
    for column in ["discount_depth_model", "discount_depth_sq"]:
        if column in counterfactual.columns:
            counterfactual[column] = 0.0

    block = prediction_data[ROLLING_OUTPUT_COLUMNS].copy()
    block["mu_hat"] = np.clip(result.predict(prediction_data), 1e-12, None)
    block["mu_hat_regular_counterfactual"] = np.clip(
        result.predict(counterfactual),
        1e-12,
        None,
    )
    block["counterfactual_model_unit_price"] = regular_price.to_numpy()
    block["counterfactual_promotion_indicator"] = 0
    block["counterfactual_post_promotion_indicator"] = 0
    block["model"] = model_name
    block["refit_block"] = block_number
    block["fit_last_week"] = fit_last_week
    block["spline_lower"] = lower
    block["spline_upper"] = upper
    block["converged"] = _result_converged(result)
    return block


def expanding_window_predictions(
    frame: pd.DataFrame,
    forecast_week_values: np.ndarray,
    model_name: str,
    refit_every_n_weeks: int = 4,
    common_controls: str = DEFAULT_COMMON_CONTROLS,
) -> pd.DataFrame:
    """Generate chronological out-of-sample predictions with periodic refits."""
    prediction_blocks: list[pd.DataFrame] = []
    start_time = perf_counter()

    for block_number, block_weeks in _forecast_blocks(
        forecast_week_values,
        refit_every_n_weeks,
    ):
        fitted = _fit_rolling_block(
            frame,
            block_weeks,
            model_name,
            common_controls,
        )
        if fitted is None:
            continue
        result, fit_data, prediction_data, lower, upper = fitted
        print(
            f"Fitted {model_name} | forecast weeks {int(block_weeks.min())}–"
            f"{int(block_weeks.max())} | training rows {len(fit_data):,}"
        )
        block = _prediction_block(
            result,
            prediction_data,
            model_name=model_name,
            block_number=block_number,
            fit_last_week=int(fit_data["week"].max()),
            lower=lower,
            upper=upper,
        )
        prediction_blocks.append(block)
        elapsed_minutes = (perf_counter() - start_time) / 60.0
        print(
            f"{model_name} | block {block_number + 1} | "
            f"predicted {int(block_weeks[0])}–{int(block_weeks[-1])} | "
            f"elapsed {elapsed_minutes:.1f} min"
        )

    if not prediction_blocks:
        raise RuntimeError(f"No predictions generated for {model_name}.")
    return pd.concat(prediction_blocks, ignore_index=True)


def common_origin_predictions(
    frame: pd.DataFrame,
    forecast_week_values: np.ndarray,
    model_name: str,
    *,
    planning_origin_week: int,
    common_controls: str = POLICY_COMMON_CONTROLS,
) -> pd.DataFrame:
    """Fit once before a planning episode and score its full future horizon."""
    weeks = np.asarray(sorted({int(value) for value in forecast_week_values}))
    if len(weeks) == 0:
        raise ValueError("forecast_week_values must contain at least one week.")
    if int(weeks.min()) != int(planning_origin_week):
        raise ValueError("planning_origin_week must be the first forecast week.")
    fit_data = frame.loc[frame["week"] < planning_origin_week].copy()
    prediction_data = frame.loc[frame["week"].isin(weeks)].copy()
    if fit_data.empty or prediction_data.empty:
        raise ValueError("Common-origin fit or prediction data are empty.")
    fit_data, prediction_data, lower, upper = prepare_price_for_spline(
        fit_data, prediction_data
    )
    formula = build_formula(model_name, lower, upper, common_controls)
    result = _fit_ppml(fit_data, formula, model_name, weeks)
    return _prediction_block(
        result,
        prediction_data,
        model_name=model_name,
        block_number=0,
        fit_last_week=int(fit_data["week"].max()),
        lower=lower,
        upper=upper,
    )


_EX_ANTE_CALENDAR_COLUMNS = (
    "calendar_month",
    "scaled_time_trend",
    "thanksgiving_week",
    "christmas_week",
    "new_year_week",
    "easter_week",
)


def build_common_origin_ex_ante_frame(
    frame: pd.DataFrame,
    forecast_week_values: np.ndarray,
    *,
    planning_origin_week: int,
    calendar_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a complete forecast grid using only origin-known panel inputs.

    Each panel's last pre-origin regular price and row-quality indicator are
    frozen across the horizon.  Future rows contribute calendar covariates only;
    their sales, realized prices, promotion states, missingness, and availability
    never enter the constructed grid.
    """
    weeks = np.asarray(sorted({int(value) for value in forecast_week_values}))
    if not len(weeks) or int(weeks.min()) != int(planning_origin_week):
        raise ValueError("Forecast weeks must begin at planning_origin_week.")
    required_origin = {
        "store", "upc", "store_upc", "week", "regular_price",
        "price_imputed_indicator",
    }
    missing = required_origin.difference(frame.columns)
    if missing:
        raise ValueError(f"Origin frame is missing {sorted(missing)}")
    origin = frame.loc[pd.to_numeric(frame["week"], errors="coerce") < planning_origin_week].copy()
    if origin.empty:
        raise ValueError("No rows precede the planning origin.")
    origin = origin.sort_values(["store_upc", "week"], kind="mergesort")
    panels = origin.groupby("store_upc", observed=True, as_index=False).tail(1).copy()
    if panels["regular_price"].isna().any() or (panels["regular_price"] <= 0).any():
        raise ValueError("Origin-known regular prices must be finite and positive.")
    panels = panels.drop(columns=["week", *_EX_ANTE_CALENDAR_COLUMNS], errors="ignore")

    calendar_source = frame if calendar_frame is None else calendar_frame
    calendar_missing = {"week", *_EX_ANTE_CALENDAR_COLUMNS}.difference(calendar_source.columns)
    if calendar_missing:
        raise ValueError(f"Calendar frame is missing {sorted(calendar_missing)}")
    calendar = calendar_source.loc[
        pd.to_numeric(calendar_source["week"], errors="coerce").isin(weeks),
        ["week", *_EX_ANTE_CALENDAR_COLUMNS],
    ].copy()
    calendar = calendar.drop_duplicates()
    inconsistent = calendar.groupby("week", observed=True).nunique(dropna=False).gt(1).any(axis=1)
    if inconsistent.any():
        raise ValueError("Calendar covariates must be unique within each forecast week.")
    calendar = calendar.drop_duplicates("week").sort_values("week")
    if set(calendar["week"]) != set(weeks):
        raise ValueError("Calendar frame does not cover every forecast week.")

    panels["_cross"] = 1
    calendar["_cross"] = 1
    output = panels.merge(calendar, on="_cross").drop(columns="_cross")
    output["week"] = output["week"].astype(int)
    output["model_unit_price"] = output["regular_price"].astype(float)
    output["unit_price_observed"] = output["regular_price"].astype(float)
    output["log_price_model"] = np.log(output["regular_price"].astype(float))
    output["promotion_indicator"] = 0
    output["post_promotion_indicator"] = 0
    output["discount_depth_model"] = 0.0
    output["discount_depth_sq"] = 0.0
    output["pricing_state"] = "regular"
    output["reference_price"] = output["regular_price"].astype(float)
    output["price_imputed"] = output["price_imputed_indicator"].astype(bool)
    output["move"] = np.nan
    if "week_start" not in output:
        output["week_start"] = pd.NaT
    return output


def common_origin_ex_ante_predictions(
    frame: pd.DataFrame,
    forecast_week_values: np.ndarray,
    model_name: str,
    *,
    planning_origin_week: int,
    calendar_frame: pd.DataFrame | None = None,
    common_controls: str = POLICY_COMMON_CONTROLS,
) -> pd.DataFrame:
    """Fit once at the origin and predict a fixed, complete ex-ante grid."""
    prediction_data = build_common_origin_ex_ante_frame(
        frame, forecast_week_values, planning_origin_week=planning_origin_week,
        calendar_frame=calendar_frame,
    )
    fit_data = frame.loc[pd.to_numeric(frame["week"], errors="coerce") < planning_origin_week].copy()
    fit_data, prediction_data, lower, upper = prepare_price_for_spline(fit_data, prediction_data)
    formula = build_formula(model_name, lower, upper, common_controls)
    result = _fit_ppml(fit_data, formula, model_name, np.asarray(forecast_week_values))
    output = _prediction_block(result, prediction_data, model_name=model_name, block_number=0,
                               fit_last_week=int(fit_data["week"].max()), lower=lower, upper=upper)
    output["prediction_design"] = "ex_ante_fixed_grid"
    output["origin_predictors_verified"] = True
    return output


def regular_counterfactual_audit(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    split: str = "test",
) -> pd.DataFrame:
    """Return promotion-state rows needed to audit PPML baseline inputs."""
    required = {
        "model",
        "split",
        "promotion_indicator",
        "post_promotion_indicator",
        "model_unit_price",
        "regular_price",
        "counterfactual_model_unit_price",
        "mu_hat",
        "mu_hat_regular_counterfactual",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Predictions lack counterfactual baseline columns: "
            f"{sorted(missing)}"
        )
    rows = predictions.loc[
        predictions["model"].astype(str).eq(model_name)
        & predictions["split"].astype(str).str.lower().eq(split.lower())
        & (
            pd.to_numeric(predictions["promotion_indicator"], errors="coerce")
            .fillna(0)
            .eq(1)
            | pd.to_numeric(
                predictions["post_promotion_indicator"],
                errors="coerce",
            )
            .fillna(0)
            .eq(1)
        ),
        [
            "store",
            "upc",
            "store_upc",
            "week",
            "promotion_indicator",
            "post_promotion_indicator",
            "model_unit_price",
            "regular_price",
            "counterfactual_model_unit_price",
            "mu_hat",
            "mu_hat_regular_counterfactual",
        ],
    ].copy()
    rows["counterfactual_promotion_indicator"] = 0
    rows["counterfactual_post_promotion_indicator"] = 0
    return rows.reset_index(drop=True)


def export_rolling_ppml_models(
    model_data: pd.DataFrame,
    forecast_weeks: np.ndarray,
    model_name: str,
    output_dir: Path,
    *,
    refit_every_n_weeks: int = 4,
    common_controls: str = DEFAULT_COMMON_CONTROLS,
) -> pd.DataFrame:
    """Save coefficient-level rolling-PPML audit artifacts.

    The ``.npz`` files deliberately contain only coefficients and recovered panel
    scales.  They are *not* reloadable prediction models: reproducing a Patsy
    prediction requires the fitted design specification (including its encoded
    categorical levels and spline construction), which is intentionally not
    serialized here.  Production counterfactual predictions are therefore
    generated in memory by :func:`_prediction_block` and persisted as prediction
    tables, rather than reconstructed from these coefficient audit artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_rows: list[dict[str, object]] = []

    for block_number, block_weeks in _forecast_blocks(
        forecast_weeks,
        refit_every_n_weeks,
    ):
        fitted = _fit_rolling_block(
            model_data,
            block_weeks,
            model_name,
            common_controls,
        )
        if fitted is None:
            continue
        result, fit_data, _, lower, upper = fitted
        model_filename = f"{model_name}_block_{block_number:03d}.npz"
        np.savez_compressed(
            output_dir / model_filename,
            params=result.params,
            panel_ids=result.panel_scales.index.to_numpy(),
            panel_scales=result.panel_scales.to_numpy(),
        )
        artifact_rows.append(
            {
                "model": model_name,
                "block_number": block_number,
                "first_forecast_week": int(block_weeks.min()),
                "last_forecast_week": int(block_weeks.max()),
                "fit_last_week": int(fit_data["week"].max()),
                "spline_lower": float(lower),
                "spline_upper": float(upper),
                "model_filename": model_filename,
                "converged": _result_converged(result),
            }
        )
    return pd.DataFrame(artifact_rows)


__all__ = [
    "DEFAULT_COMMON_CONTROLS",
    "POLICY_COMMON_CONTROLS",
    "build_formula",
    "build_product_selection_table",
    "common_origin_predictions",
    "common_origin_ex_ante_predictions",
    "build_common_origin_ex_ante_frame",
    "expanding_window_predictions",
    "export_rolling_ppml_models",
    "panel_period_counts",
    "period_availability",
    "paired_weekly_loss_test",
    "poisson_deviance_contribution",
    "prediction_metrics",
    "prepare_price_for_spline",
    "regular_counterfactual_audit",
    "spline_expression",
]
