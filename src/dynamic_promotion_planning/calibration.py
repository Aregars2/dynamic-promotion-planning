"""Behavioral calibration for product-level promotion dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .promotion import canonical_promotion_indicator

PARAMETER_COLUMNS = [
    "price_elasticity",
    "promotion_lift_log",
    "displacement_strength",
    "inventory_persistence",
]

@dataclass(frozen=True)
class CalibrationConfig:
    primary_model: str = "product_promotion"
    max_products: int = 8
    bootstrap_replications: int = 1_000
    random_seed: int = 42
    event_window: tuple[int, ...] = tuple(range(-4, 7))
    holdout_window: tuple[int, ...] = tuple(range(-3, 5))
    pre_event_weeks: tuple[int, ...] = (-3, -2, -1)
    post_lags: tuple[int, ...] = (1, 2, 3, 4)
    min_isolated_events: int = 15
    min_depth_events: int = 10
    min_common_panels: int = 4
    min_calibration_discount: float = 0.05
    persistence_grid: tuple[float, ...] = (0.15, 0.35, 0.60)
    min_persistence_signal: float = 0.05
    max_persistence_slope_ratio: float = 1.25
    action_bandwidth: float = 0.025
    well_supported_action_count: int = 20

def first_existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            f"None of these columns were found: {candidates}"
        )

    return None

def weighted_mean(
    values: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)

    if not valid.any():
        return np.nan

    return float(
        np.average(
            x[valid],
            weights=w[valid],
        )
    )

def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)

    if not valid.any():
        return np.nan

    x = x[valid]
    w = w[valid]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w) / np.sum(w)

    return float(
        np.interp(
            quantile,
            cumulative,
            x,
        )
    )

def weighted_slope(
    x: pd.Series,
    y: pd.Series,
    weights: pd.Series,
) -> float:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    wv = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(xv)
        & np.isfinite(yv)
        & np.isfinite(wv)
        & (wv > 0)
    )

    if valid.sum() < 3:
        return np.nan

    xv = xv[valid]
    yv = yv[valid]
    wv = wv[valid]
    x_mean = np.average(xv, weights=wv)
    y_mean = np.average(yv, weights=wv)
    denominator = np.sum(wv * (xv - x_mean) ** 2)

    if denominator <= 1e-12:
        return np.nan

    numerator = np.sum(
        wv
        * (xv - x_mean)
        * (yv - y_mean)
    )

    return float(numerator / denominator)

def alternating_residual(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
    max_iterations: int = 300,
    tolerance: float = 1e-10,
) -> pd.Series:
    values = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    ).astype(float)

    if values.isna().any():
        raise ValueError(
            f"{value_column!r} contains missing values."
        )

    residual = values - values.mean()

    for _ in range(max_iterations):
        previous = residual.to_numpy(copy=True)

        for group_column in group_columns:
            residual = (
                residual
                - residual.groupby(
                    frame[group_column],
                    observed=True,
                ).transform("mean")
            )

        change = float(
            np.max(
                np.abs(
                    residual.to_numpy() - previous
                )
            )
        )

        if change < tolerance:
            break

    return residual

def isolated_promotion_mask(
    frame: pd.DataFrame,
    pre_gap: int,
    post_gap: int,
) -> pd.Series:
    grouped = frame.groupby(
        "store_upc",
        observed=True,
        sort=False,
    )
    mask = frame["promotion_indicator"].eq(1)

    for offset in range(1, pre_gap + 1):
        previous_promo = grouped[
            "promotion_indicator"
        ].shift(offset)
        previous_week = grouped["week"].shift(offset)

        mask &= (
            previous_promo.eq(0)
            & frame["week"].sub(previous_week).eq(offset)
        )

    for offset in range(1, post_gap + 1):
        next_promo = grouped[
            "promotion_indicator"
        ].shift(-offset)
        next_week = grouped["week"].shift(-offset)

        mask &= (
            next_promo.eq(0)
            & next_week.sub(frame["week"]).eq(offset)
        )

    return mask.fillna(False)

def _load_panel_selection(
    history_raw: pd.DataFrame,
    prediction_path: Path,
    primary_model: str,
) -> tuple[set[str], int, int]:
    if prediction_path.is_file():
        predictions = pd.read_pickle(
            prediction_path
        ).copy()
        required = {
            "store_upc",
            "week",
            "model",
            "split",
        }
        missing = required.difference(
            predictions.columns
        )

        if missing:
            raise ValueError(
                f"Prediction artifact missing: {sorted(missing)}"
            )

        predictions["store_upc"] = (
            predictions["store_upc"].astype(str)
        )
        predictions["week"] = pd.to_numeric(
            predictions["week"],
            errors="coerce",
        )

        primary = predictions.loc[
            predictions["model"].astype(str).eq(
                primary_model
            )
        ].copy()

        selected_panels = set(
            primary["store_upc"].dropna().unique()
        )

        calibration_weeks = primary.loc[
            primary["split"].astype(str).eq(
                "calibration"
            ),
            "week",
        ].dropna()

        evaluation_weeks = primary.loc[
            primary["split"].astype(str).eq(
                "test"
            ),
            "week",
        ].dropna()

        if calibration_weeks.empty:
            raise ValueError(
                "No calibration weeks in prediction artifact."
            )

        calibration_end = int(
            calibration_weeks.max()
        )
        evaluation_start = (
            int(evaluation_weeks.min())
            if not evaluation_weeks.empty
            else calibration_end + 1
        )

        return (
            selected_panels,
            calibration_end,
            evaluation_start,
        )

    panel_counts = (
        history_raw.assign(
            store_upc=history_raw["store_upc"].astype(str)
        )
        .groupby(
            "store_upc",
            observed=True,
        )
        .size()
        .sort_values(
            ascending=False
        )
    )
    selected_panels = set(
        panel_counts.head(120).index
    )
    week_values = pd.to_numeric(
        history_raw["week"],
        errors="coerce",
    ).dropna()
    calibration_end = int(
        np.quantile(
            week_values,
            0.80,
        )
    )

    warnings.warn(
        "Prediction artifact missing; using the longest panels "
        "and an 80% chronological split."
    )

    return (
        selected_panels,
        calibration_end,
        calibration_end + 1,
    )

def _normalize_history_identifiers(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    for column in ["store", "upc", "store_upc"]:
        history[column] = history[column].astype(str)
    history["week"] = pd.to_numeric(history["week"], errors="coerce")
    history["move"] = pd.to_numeric(history["move"], errors="coerce")
    return history


def _construct_price_variables(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    price_column = first_existing_column(
        history,
        ["model_unit_price", "unit_price_observed", "unit_price", "price"],
    )
    history["price"] = pd.to_numeric(history[price_column], errors="coerce")

    regular_price_column = first_existing_column(
        history,
        ["regular_price", "reference_price"],
        required=False,
    )
    if regular_price_column is None:
        history["regular_price_model"] = history.groupby(
            "store_upc",
            observed=True,
        )["price"].transform(
            lambda values: values.rolling(13, min_periods=4).max()
        )
    else:
        history["regular_price_model"] = pd.to_numeric(
            history[regular_price_column],
            errors="coerce",
        )
    history["regular_price_model"] = (
        history["regular_price_model"]
        .where(history["regular_price_model"].gt(0))
        .fillna(history["price"])
    )

    discount_column = first_existing_column(
        history,
        ["discount_depth_model", "discount_depth"],
        required=False,
    )
    if discount_column is None:
        history["discount_depth_model"] = (
            1.0 - history["price"] / history["regular_price_model"]
        )
    else:
        history["discount_depth_model"] = pd.to_numeric(
            history[discount_column],
            errors="coerce",
        )
    history["discount_depth_model"] = history[
        "discount_depth_model"
    ].fillna(0.0).clip(0.0, 0.80)
    return history


def _construct_promotion_indicators(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    promotion_column = first_existing_column(
        history,
        ["promotion_indicator", "promo_recorded", "promo_from_discount"],
        required=False,
    )
    history["promotion_indicator"] = canonical_promotion_indicator(
        history,
        recorded_column=promotion_column,
    )

    post_column = first_existing_column(
        history,
        ["post_promotion_indicator", "post_promo"],
        required=False,
    )
    if post_column is None:
        history["post_promotion_indicator"] = (
            history.groupby("store_upc", observed=True)["promotion_indicator"]
            .shift(1)
            .fillna(0)
            .astype(int)
        )
    else:
        history["post_promotion_indicator"] = (
            pd.to_numeric(history[post_column], errors="coerce")
            .fillna(0)
            .gt(0)
            .astype(int)
        )
    return history


def _filter_valid_history(history: pd.DataFrame) -> pd.DataFrame:
    if "price_imputed" in history.columns:
        price_imputed = history["price_imputed"].fillna(False).astype(bool)
    else:
        price_imputed = pd.Series(False, index=history.index)
    valid = (
        history["week"].notna()
        & history["move"].notna()
        & np.isfinite(history["move"])
        & history["move"].ge(0)
        & history["price"].notna()
        & np.isfinite(history["price"])
        & history["price"].gt(0)
        & ~price_imputed
    )
    filtered = history.loc[valid].copy()
    filtered["week"] = filtered["week"].astype(int)
    if filtered.duplicated(["store_upc", "week"]).any():
        raise ValueError("Duplicate store-product-week rows found.")
    filtered["log1p_move"] = np.log1p(filtered["move"])
    filtered["log_price"] = np.log(filtered["price"])
    return filtered


def _construct_observed_cost(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    margin_column = "gross_margin_pct_observed"
    if margin_column not in history.columns:
        raise KeyError(f"Required margin column {margin_column!r} missing.")
    raw_margin = pd.to_numeric(history[margin_column], errors="coerce")
    finite = raw_margin.loc[raw_margin.notna() & np.isfinite(raw_margin)]
    margin_scale = (
        100.0
        if not finite.empty and finite.abs().quantile(0.95) > 1.5
        else 1.0
    )
    history["gross_margin_fraction"] = raw_margin / margin_scale
    valid_margin = history["gross_margin_fraction"].between(
        -1.0,
        0.95,
        inclusive="both",
    )
    history["unit_cost_observed"] = history["price"] * (
        1.0 - history["gross_margin_fraction"]
    )
    invalid_cost = (
        ~valid_margin
        | ~np.isfinite(history["unit_cost_observed"])
        | history["unit_cost_observed"].le(0)
    )
    history.loc[invalid_cost, "unit_cost_observed"] = np.nan
    return history


def _select_calibration_products(
    history: pd.DataFrame,
    maximum_products: int,
) -> tuple[pd.DataFrame, list[str]]:
    coverage = (
        history.groupby("upc", observed=True)
        .agg(
            panels=("store_upc", "nunique"),
            rows=("week", "size"),
            promotion_rows=("promotion_indicator", "sum"),
        )
        .reset_index()
    )
    coverage["selection_score"] = coverage["panels"] * np.log1p(
        coverage["promotion_rows"]
    )
    selected = (
        coverage.sort_values(
            ["selection_score", "rows"],
            ascending=False,
        )
        .head(maximum_products)["upc"]
        .astype(str)
        .tolist()
    )
    return coverage, selected


def _product_name_lookup(
    history: pd.DataFrame,
    selected_products: list[str],
) -> dict[str, str]:
    description_column = first_existing_column(
        history,
        ["descrip", "description", "product_description"],
        required=False,
    )
    if description_column is None:
        return {upc: upc for upc in selected_products}
    return (
        history.groupby("upc", observed=True)[description_column]
        .agg(
            lambda values: (
                values.dropna().astype(str).iloc[0]
                if not values.dropna().empty
                else ""
            )
        )
        .to_dict()
    )


def load_clean_history(
    data_path: Path,
    prediction_path: Path,
    config: CalibrationConfig,
) -> dict[str, Any]:
    """Load the selected panel history and construct calibration variables."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Demand data not found: {data_path}")
    history_raw = pd.read_parquet(data_path)
    selected_panels, calibration_end, evaluation_start = _load_panel_selection(
        history_raw,
        prediction_path,
        config.primary_model,
    )

    history = _normalize_history_identifiers(history_raw)
    history = history.loc[history["store_upc"].isin(selected_panels)].copy()
    history = _construct_price_variables(history)
    history = _construct_promotion_indicators(history)
    history = _filter_valid_history(history)
    history = _construct_observed_cost(history)
    history["sample_period"] = np.where(
        history["week"].le(calibration_end),
        "calibration",
        "evaluation",
    )

    coverage, selected_products = _select_calibration_products(
        history,
        config.max_products,
    )
    history = history.loc[history["upc"].isin(selected_products)].copy()
    product_names = _product_name_lookup(history, selected_products)
    history = history.sort_values(
        ["upc", "store_upc", "week"]
    ).reset_index(drop=True)
    return {
        "history": history,
        "selected_products": selected_products,
        "product_names": product_names,
        "calibration_end_week": calibration_end,
        "evaluation_start_week": evaluation_start,
        "coverage": coverage,
    }

def prepare_product_period(
    product_frame: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    output = product_frame.loc[
        product_frame["sample_period"].eq(
            period
        )
    ].copy()

    if output.empty:
        return output

    output["demand_residual"] = alternating_residual(
        output,
        "log1p_move",
        [
            "store_upc",
            "week",
        ],
    )
    output["price_residual"] = alternating_residual(
        output,
        "log_price",
        [
            "store_upc",
            "week",
        ],
    )

    return output

def _select_isolated_promotion_events(
    product_period: pd.DataFrame,
    minimum_events: int,
) -> tuple[pd.DataFrame, tuple[int, int]]:
    selected_gap = (3, 3)
    selected_mask = isolated_promotion_mask(
        product_period,
        pre_gap=3,
        post_gap=3,
    )
    if int(selected_mask.sum()) < minimum_events:
        selected_mask = pd.Series(False, index=product_period.index)

    events = (
        product_period.loc[
            selected_mask,
            ["upc", "store", "store_upc", "week", "discount_depth_model"],
        ]
        .copy()
        .rename(
            columns={
                "week": "event_week",
                "discount_depth_model": "event_discount_depth",
            }
        )
    )
    events["event_id"] = np.arange(len(events), dtype=int)
    return events, selected_gap


def _expand_event_window(
    product_period: pd.DataFrame,
    events: pd.DataFrame,
    event_window: tuple[int, ...],
) -> pd.DataFrame:
    source = product_period[
        ["store_upc", "week", "demand_residual", "price_residual", "move"]
    ].copy()
    blocks: list[pd.DataFrame] = []
    anchor_columns = [
        "event_id",
        "upc",
        "store_upc",
        "event_week",
        "event_discount_depth",
    ]
    for relative_week in event_window:
        block = events[anchor_columns].copy()
        block["relative_week"] = int(relative_week)
        block["week"] = block["event_week"] + int(relative_week)
        blocks.append(
            block.merge(
                source,
                on=["store_upc", "week"],
                how="left",
                validate="many_to_one",
            )
        )
    return pd.concat(blocks, ignore_index=True)


def _add_event_pre_period_baseline(
    event_long: pd.DataFrame,
    pre_event_weeks: tuple[int, ...],
) -> pd.DataFrame:
    pre_means = (
        event_long.loc[event_long["relative_week"].isin(pre_event_weeks)]
        .groupby("event_id", observed=True)
        .agg(
            pre_demand_mean=("demand_residual", "mean"),
            pre_price_mean=("price_residual", "mean"),
            pre_observations=("demand_residual", "count"),
        )
        .reset_index()
    )
    event_long = event_long.merge(
        pre_means,
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    event_long = event_long.loc[event_long["pre_observations"].ge(2)].copy()
    event_long["event_effect"] = (
        event_long["demand_residual"] - event_long["pre_demand_mean"]
    )
    event_long["event_price_change"] = (
        event_long["price_residual"] - event_long["pre_price_mean"]
    )
    return event_long


def _event_level_summary(event_long: pd.DataFrame) -> pd.DataFrame:
    summary = (
        event_long.pivot_table(
            index=[
                "event_id",
                "upc",
                "store_upc",
                "event_discount_depth",
            ],
            columns="relative_week",
            values="event_effect",
            aggfunc="first",
        )
        .reset_index()
    )
    current_price = (
        event_long.loc[
            event_long["relative_week"].eq(0),
            ["event_id", "event_price_change"],
        ]
        .drop_duplicates("event_id")
        .rename(columns={"event_price_change": "current_price_change"})
    )
    summary = summary.merge(
        current_price,
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    required = [
        column
        for column in [0, 1, "current_price_change"]
        if column in summary.columns
    ]
    summary = summary.dropna(subset=required).copy()
    if 0 in summary.columns:
        summary = summary.rename(columns={0: "current_lift_log"})
    else:
        summary["current_lift_log"] = np.nan
    for lag in [1, 2, 3, 4]:
        summary[f"post{lag}_dip_log"] = (
            -summary[lag] if lag in summary.columns else np.nan
        )
    return summary


def build_event_data(
    product_period: pd.DataFrame,
    event_window: tuple[int, ...],
    pre_event_weeks: tuple[int, ...],
    minimum_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, int]]:
    """Construct isolated promotion-event panels and event-level moments."""
    events, selected_gap = _select_isolated_promotion_events(
        product_period,
        minimum_events,
    )
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), selected_gap
    event_long = _expand_event_window(product_period, events, event_window)
    event_long = _add_event_pre_period_baseline(event_long, pre_event_weeks)
    event_summary = _event_level_summary(event_long)
    return event_long, event_summary, selected_gap

def _bootstrap_product(
    depth_events: pd.DataFrame,
    panel_price: pd.DataFrame,
    common_panels: np.ndarray,
    config: CalibrationConfig,
    random_seed: int,
) -> pd.DataFrame:
    if (
        len(common_panels)
        < config.min_common_panels
        or len(depth_events)
        < config.min_depth_events
    ):
        return pd.DataFrame()

    event_source = depth_events.loc[
        depth_events["store_upc"].isin(
            common_panels
        )
    ].copy()
    price_source = panel_price.loc[
        panel_price["store_upc"].isin(
            common_panels
        )
    ].copy()
    rng = np.random.default_rng(
        random_seed
    )
    rows = []

    for bootstrap_id in range(
        config.bootstrap_replications
    ):
        sampled_panels = rng.choice(
            common_panels,
            size=len(common_panels),
            replace=True,
        )
        counts = (
            pd.Series(sampled_panels)
            .value_counts()
            .rename("cluster_weight")
        )
        events_boot = event_source.merge(
            counts,
            left_on="store_upc",
            right_index=True,
            how="inner",
            validate="many_to_one",
        )
        price_boot = price_source.merge(
            counts,
            left_on="store_upc",
            right_index=True,
            how="inner",
            validate="one_to_one",
        )
        event_weights = events_boot[
            "cluster_weight"
        ].astype(float)
        price_weights = price_boot[
            "cluster_weight"
        ].astype(float)
        denominator = float(
            np.sum(
                price_weights
                * price_boot["xx"]
            )
        )
        numerator = float(
            np.sum(
                price_weights
                * price_boot["xy"]
            )
        )
        log_price_coefficient = (
            numerator / denominator
            if denominator > 1e-12
            else np.nan
        )
        elasticity = float(
            np.clip(
                -log_price_coefficient,
                0.10,
                5.00,
            )
        )
        ordinary_price_effect = (
            log_price_coefficient
            * events_boot[
                "current_price_change"
            ]
        )
        promotion_lift = (
            events_boot[
                "current_lift_log"
            ]
            - ordinary_price_effect
        )
        row = {
            "bootstrap_id": bootstrap_id,
            "price_elasticity": elasticity,
            "promotion_lift_log": weighted_mean(
                promotion_lift,
                event_weights,
            ),
        }

        for lag in config.post_lags:
            row[
                f"post{lag}_depth_slope"
            ] = weighted_slope(
                events_boot[
                    "event_discount_depth"
                ],
                events_boot[
                    f"post{lag}_dip_log"
                ],
                event_weights,
            )

        rows.append(row)

    return pd.DataFrame(rows)

def _pooled_sampling_inputs(
    pooled_draws: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    pooled_valid = pooled_draws.dropna(subset=PARAMETER_COLUMNS).copy()
    if pooled_valid.empty:
        raise ValueError("No complete pooled behavioral draws are available.")
    if "draw_weight" in pooled_valid.columns:
        probabilities = pooled_valid["draw_weight"].to_numpy(dtype=float)
    else:
        probabilities = np.ones(len(pooled_valid), dtype=float)
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise ValueError("Pooled draw weights are invalid.")
    return pooled_valid, probabilities / probabilities.sum()


def _event_study_summary(
    event_long: pd.DataFrame,
    upc: str,
    product_name: str,
) -> pd.DataFrame | None:
    if event_long.empty:
        return None
    summary = (
        event_long.groupby("relative_week", observed=True)
        .agg(
            observed_effect=("event_effect", "mean"),
            events=("event_id", "nunique"),
        )
        .reset_index()
    )
    summary["upc"] = upc
    summary["product_name"] = product_name
    return summary


def _select_depth_events(
    event_summary: pd.DataFrame,
    config: CalibrationConfig,
) -> tuple[pd.DataFrame, float]:
    calibration_discount = config.min_calibration_discount
    if event_summary.empty:
        return pd.DataFrame(), calibration_discount
    depth_events = event_summary.loc[
        event_summary["event_discount_depth"].ge(calibration_discount)
    ].copy()
    if len(depth_events) < config.min_depth_events:
        return pd.DataFrame(), calibration_discount
    return depth_events, calibration_discount


def _panel_price_moments(calibration: pd.DataFrame) -> pd.DataFrame:
    regular = calibration.loc[
        calibration["promotion_indicator"].eq(0)
        & calibration["post_promotion_indicator"].eq(0)
    ].copy()
    return (
        regular.assign(
            xx=lambda frame: frame["price_residual"] ** 2,
            xy=lambda frame: frame["price_residual"] * frame["demand_residual"],
        )
        .groupby("store_upc", observed=True)
        .agg(
            xx=("xx", "sum"),
            xy=("xy", "sum"),
            observations=("week", "size"),
        )
        .reset_index()
    )


def _sample_pooled_draws(
    pooled_valid: pd.DataFrame,
    probabilities: np.ndarray,
    rng: np.random.Generator,
    replications: int,
) -> pd.DataFrame:
    indices = rng.choice(
        len(pooled_valid),
        size=replications,
        replace=True,
        p=probabilities,
    )
    return pooled_valid.iloc[indices][PARAMETER_COLUMNS].reset_index(drop=True)


def _legacy_heuristic_reliability(
    depth_event_count: int,
    common_panel_count: int,
    config: CalibrationConfig,
) -> float:
    """Return the retired support-count heuristic for diagnostics only."""
    del config
    event_reliability = (
        depth_event_count / (depth_event_count + 30.0)
        if depth_event_count > 0
        else 0.0
    )
    panel_reliability = (
        common_panel_count / (common_panel_count + 6.0)
        if common_panel_count > 0
        else 0.0
    )
    return float(np.clip(event_reliability * panel_reliability, 0.0, 1.0))


EB_PARAMETERS: dict[str, tuple[str, float, float]] = {
    "price_elasticity": ("price_elasticity", 0.10, 5.00),
    "promotion_lift_log": ("promotion_lift_log", -1.00, 3.00),
    "displacement_strength": ("post1_depth_slope", 0.00, 3.00),
}


def _raw_product_parameter_draws(
    bootstrap: pd.DataFrame,
    pooled_sample: pd.DataFrame,
    config: CalibrationConfig,
) -> pd.DataFrame:
    """Return policy-scale product bootstrap draws, retaining bootstrap IDs."""
    if bootstrap.empty:
        product_base = pooled_sample.copy()
        product_base["product_calibration_source"] = "pooled_fallback"
        product_base["bootstrap_id"] = np.arange(config.bootstrap_replications)
        product_base["post1_depth_slope"] = product_base[
            "displacement_strength"
        ]
        product_base["post2_depth_slope"] = (
            product_base["displacement_strength"]
            * product_base["inventory_persistence"]
        )
    else:
        product_base = bootstrap.sort_values("bootstrap_id").reset_index(drop=True).copy()
        if len(product_base) != config.bootstrap_replications:
            raise ValueError(
                "Product bootstrap does not contain the configured number of draws."
            )
        product_base["product_calibration_source"] = "product_bootstrap"
        product_base["inventory_persistence"] = pooled_sample[
            "inventory_persistence"
        ]

    for parameter, (source, lower, upper) in EB_PARAMETERS.items():
        if source not in product_base:
            product_base[source] = np.nan
        fallback = pooled_sample[parameter].to_numpy(dtype=float)
        raw = product_base[source].to_numpy(dtype=float)
        product_base[parameter] = np.clip(
            np.where(np.isfinite(raw), raw, fallback), lower, upper
        )
    return product_base


def _empirical_bayes_weights(
    raw_draws_by_product: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, float]], pd.DataFrame, pd.DataFrame]:
    """Estimate parameter-specific EB weights from policy-scale bootstrap draws."""
    products = sorted(raw_draws_by_product)
    weights: dict[str, dict[str, float]] = {upc: {} for upc in products}
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for parameter in EB_PARAMETERS:
        means = np.array([
            raw_draws_by_product[upc][parameter].mean() for upc in products
        ], dtype=float)
        within_variances = np.array([
            raw_draws_by_product[upc][parameter].var(ddof=1) for upc in products
        ], dtype=float)
        observed_variance = float(np.var(means, ddof=1)) if len(products) > 1 else 0.0
        mean_within_variance = float(np.mean(within_variances))
        tau2 = float(max(0.0, observed_variance - mean_within_variance))
        parameter_weights = (
            np.zeros_like(within_variances)
            if tau2 == 0.0
            else tau2 / (tau2 + within_variances)
        )
        for upc, theta_hat, sampling_variance, weight in zip(
            products, means, within_variances, parameter_weights, strict=True
        ):
            weight_float = float(weight)
            weights[upc][parameter] = weight_float
            detail_rows.append(
                {
                    "upc": upc,
                    "parameter": parameter,
                    "theta_hat": float(theta_hat),
                    "bootstrap_variance": float(sampling_variance),
                    "bootstrap_standard_error": float(np.sqrt(sampling_variance)),
                    "tau2": tau2,
                    "eb_weight": weight_float,
                }
            )
        summary_rows.append(
            {
                "parameter": parameter,
                "products": len(products),
                "observed_cross_product_variance": observed_variance,
                "mean_within_product_bootstrap_variance": mean_within_variance,
                "tau2": tau2,
                "tau2_at_zero": bool(tau2 == 0.0),
                "minimum_eb_weight": float(parameter_weights.min()),
                "median_eb_weight": float(np.median(parameter_weights)),
                "maximum_eb_weight": float(parameter_weights.max()),
            }
        )
    return weights, pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def _apply_empirical_bayes_draws(
    raw_product_draws: pd.DataFrame,
    pooled_sample: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Mix aligned product/pooled bootstrap replicate m using EB weights."""
    product_base = raw_product_draws.copy()
    for parameter in EB_PARAMETERS:
        weight = weights[parameter]
        product_base[parameter] = (
            weight * product_base[parameter].to_numpy(dtype=float)
            + (1.0 - weight) * pooled_sample[parameter].to_numpy(dtype=float)
        )
        product_base[f"eb_weight_{parameter}"] = weight
    return product_base


def _persistence_values_for_draw(
    row: dict[str, Any],
    config: CalibrationConfig,
) -> tuple[list[float], str, bool]:
    if row.get("product_calibration_source") == "pooled_fallback":
        pooled_persistence = float(np.clip(row["inventory_persistence"], 0.05, 0.95))
        return sorted({*config.persistence_grid, pooled_persistence}), "pooled_fallback_grid", False
    first_slope = row.get("post1_depth_slope", np.nan)
    second_slope = row.get("post2_depth_slope", np.nan)
    informative = bool(
        np.isfinite(first_slope)
        and np.isfinite(second_slope)
        and first_slope >= config.min_persistence_signal
        and 0 < second_slope <= config.max_persistence_slope_ratio * first_slope
    )
    if informative:
        values = [float(np.clip(second_slope / first_slope, 0.05, 0.95))]
        return values, "product_depth_decay", True

    pooled_persistence = float(
        np.clip(
            row.get("inventory_persistence", np.median(config.persistence_grid)),
            0.05,
            0.95,
        )
    )
    values = sorted({*config.persistence_grid, pooled_persistence})
    return values, "partial_id_plus_pooled", False


def _expand_persistence_draws(
    product_base: pd.DataFrame,
    config: CalibrationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for draw in product_base.itertuples(index=False):
        draw_dict = draw._asdict()
        values, source, informative = _persistence_values_for_draw(
            draw_dict,
            config,
        )
        for persistence in values:
            expanded = dict(draw_dict)
            expanded["inventory_persistence"] = float(persistence)
            expanded["persistence_source"] = source
            expanded["persistence_informative"] = informative
            expanded["draw_weight"] = 1.0 / (
                config.bootstrap_replications * len(values)
            )
            rows.append(expanded)
    return pd.DataFrame(rows)


def _product_economic_references(
    calibration: pd.DataFrame,
    pooled_draws: pd.DataFrame,
) -> tuple[float, float, float]:
    regular = calibration.loc[
        calibration["promotion_indicator"].eq(0)
        & calibration["post_promotion_indicator"].eq(0)
    ]
    base_demand = float(regular["move"].median())
    regular_price = float(regular["regular_price_model"].median())
    unit_cost = float(calibration["unit_cost_observed"].median())
    if not np.isfinite(base_demand) or base_demand <= 0:
        base_demand = float(calibration["move"].median())
    if not np.isfinite(regular_price) or regular_price <= 0:
        regular_price = float(calibration["price"].median())
    if not np.isfinite(unit_cost) or unit_cost <= 0:
        unit_cost = float(pooled_draws["unit_cost"].median())
    return base_demand, regular_price, unit_cost


def _annotate_product_draws(
    frames: tuple[pd.DataFrame, ...],
    *,
    upc: str,
    product_name: str,
    base_demand: float,
    regular_price: float,
    unit_cost: float,
    legacy_reliability: float,
) -> None:
    for frame in frames:
        frame["upc"] = upc
        frame["product_name"] = product_name
        frame["base_demand"] = base_demand
        frame["regular_price"] = regular_price
        frame["unit_cost"] = unit_cost
        frame["legacy_heuristic_reliability"] = legacy_reliability


def _product_summary_row(
    *,
    upc: str,
    product_name: str,
    product_frame: pd.DataFrame,
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    event_summary: pd.DataFrame,
    depth_events: pd.DataFrame,
    common_panels: np.ndarray,
    selected_gap: tuple[int, int],
    calibration_discount: float,
    legacy_reliability: float,
    product_draws: pd.DataFrame,
    base_demand: float,
    regular_price: float,
    unit_cost: float,
) -> dict[str, Any]:
    return {
        "upc": upc,
        "product_name": product_name,
        "calibration_rows": int(len(calibration)),
        "evaluation_rows": int(len(evaluation)),
        "panels": int(product_frame["store_upc"].nunique()),
        "isolated_events": int(len(event_summary)),
        "depth_events": int(len(depth_events)),
        "common_panels": int(len(common_panels)),
        "selected_gap": f"{selected_gap[0]}/{selected_gap[1]}",
        "calibration_discount": calibration_discount,
        "legacy_heuristic_reliability": legacy_reliability,
        "price_elasticity_median": float(
            product_draws["price_elasticity"].median()
        ),
        "promotion_lift_median": float(
            product_draws["promotion_lift_log"].median()
        ),
        "displacement_median": float(
            product_draws["displacement_strength"].median()
        ),
        "persistence_median": float(
            product_draws["inventory_persistence"].median()
        ),
        "persistence_informative_share": float(
            product_draws["persistence_informative"].mean()
        ),
        "base_demand": base_demand,
        "regular_price": regular_price,
        "unit_cost": unit_cost,
    }


def _holdout_prediction_matrix(
    relative_week: int,
    holdout_summary: pd.DataFrame,
    theta_draws: np.ndarray,
) -> np.ndarray:
    event_count = len(holdout_summary)
    if relative_week < 0:
        return np.zeros((event_count, len(theta_draws)), dtype=float)
    if relative_week == 0:
        price_changes = holdout_summary["current_price_change"].to_numpy(
            dtype=float
        )
        return (
            -theta_draws[None, :, 0] * price_changes[:, None]
            + theta_draws[None, :, 1]
        )
    depths = holdout_summary["event_discount_depth"].to_numpy(dtype=float)
    return (
        -depths[:, None]
        * theta_draws[None, :, 2]
        * np.power(theta_draws[None, :, 3], relative_week - 1)
    )


def _product_holdout_rows(
    evaluation: pd.DataFrame,
    product_draws: pd.DataFrame,
    *,
    upc: str,
    product_name: str,
    config: CalibrationConfig,
) -> list[dict[str, Any]]:
    holdout_long, holdout_summary, holdout_gap = build_event_data(
        evaluation,
        config.holdout_window,
        config.pre_event_weeks,
        minimum_events=5,
    )
    if holdout_summary.empty:
        return []

    theta_draws = product_draws[PARAMETER_COLUMNS].to_numpy(dtype=float)
    draw_weights = product_draws["draw_weight"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for relative_week in config.holdout_window:
        observed = holdout_long.loc[
            holdout_long["relative_week"].eq(relative_week),
            "event_effect",
        ]
        prediction_matrix = _holdout_prediction_matrix(
            relative_week,
            holdout_summary,
            theta_draws,
        )
        draw_predictions = prediction_matrix.mean(axis=0)
        predicted_mean = float(np.average(draw_predictions, weights=draw_weights))
        observed_mean = float(observed.mean())
        rows.append(
            {
                "upc": upc,
                "product_name": product_name,
                "relative_week": int(relative_week),
                "observed_effect": observed_mean,
                "predicted_effect": predicted_mean,
                "predicted_q10": weighted_quantile(
                    draw_predictions, draw_weights, 0.10
                ),
                "predicted_q90": weighted_quantile(
                    draw_predictions, draw_weights, 0.90
                ),
                "absolute_error": float(abs(observed_mean - predicted_mean)),
                "events": int(holdout_summary["event_id"].nunique()),
                "selected_gap": f"{holdout_gap[0]}/{holdout_gap[1]}",
            }
        )
    return rows


def _summarize_product_holdout(product_holdout: pd.DataFrame) -> pd.DataFrame:
    if product_holdout.empty:
        return pd.DataFrame()
    return (
        product_holdout.groupby(["upc", "product_name"], observed=True)
        .agg(
            holdout_events=("events", "max"),
            mean_absolute_error=("absolute_error", "mean"),
            post_period_mae=(
                "absolute_error",
                lambda values: float(values.iloc[4:].mean()),
            ),
        )
        .reset_index()
    )


def calibrate_products(
    history: pd.DataFrame,
    selected_products: list[str],
    product_names: dict[str, str],
    pooled_draws: pd.DataFrame,
    config: CalibrationConfig,
) -> dict[str, pd.DataFrame]:
    """Calibrate product-level draws with partial pooling and holdout diagnostics."""
    pooled_valid, probabilities = _pooled_sampling_inputs(pooled_draws)
    rng = np.random.default_rng(config.random_seed)
    # One pooled draw sequence is shared across products.  Bootstrap replicate m
    # therefore denotes the same pooled parent throughout the policy system.
    pooled_sample = _sample_pooled_draws(
        pooled_valid,
        probabilities,
        rng,
        config.bootstrap_replications,
    )
    draw_frames: list[pd.DataFrame] = []
    base_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    event_studies: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    raw_draws_by_product: dict[str, pd.DataFrame] = {}

    for product_index, raw_upc in enumerate(selected_products):
        upc = str(raw_upc)
        product_name = product_names.get(upc, upc)
        product_frame = history.loc[history["upc"].eq(upc)].copy()
        calibration = prepare_product_period(product_frame, "calibration")
        evaluation = prepare_product_period(product_frame, "evaluation")
        event_long, event_summary, selected_gap = build_event_data(
            calibration,
            config.event_window,
            config.pre_event_weeks,
            config.min_isolated_events,
        )
        event_study = _event_study_summary(event_long, upc, product_name)
        if event_study is not None:
            event_studies.append(event_study)

        depth_events, calibration_discount = _select_depth_events(
            event_summary,
            config,
        )
        panel_price = _panel_price_moments(calibration)
        common_panels = np.array(
            sorted(
                set(depth_events.get("store_upc", pd.Series(dtype=str)).unique())
                .intersection(set(panel_price["store_upc"].unique()))
            )
        )
        bootstrap = _bootstrap_product(
            depth_events,
            panel_price,
            common_panels,
            config,
            config.random_seed + product_index + 1,
        )
        raw_product_draws = _raw_product_parameter_draws(
            bootstrap,
            pooled_sample,
            config,
        )
        raw_draws_by_product[upc] = raw_product_draws
        contexts.append(
            {
                "upc": upc,
                "product_name": product_name,
                "product_frame": product_frame,
                "calibration": calibration,
                "evaluation": evaluation,
                "event_summary": event_summary,
                "depth_events": depth_events,
                "common_panels": common_panels,
                "selected_gap": selected_gap,
                "calibration_discount": calibration_discount,
                "raw_product_draws": raw_product_draws,
                "legacy_reliability": _legacy_heuristic_reliability(
                    len(depth_events),
                    len(common_panels),
                    config,
                ),
            }
        )

    eb_weights, eb_detail, eb_summary = _empirical_bayes_weights(
        raw_draws_by_product
    )

    for context in contexts:
        upc = context["upc"]
        product_name = context["product_name"]
        product_frame = context["product_frame"]
        calibration = context["calibration"]
        evaluation = context["evaluation"]
        event_summary = context["event_summary"]
        depth_events = context["depth_events"]
        common_panels = context["common_panels"]
        selected_gap = context["selected_gap"]
        calibration_discount = context["calibration_discount"]
        legacy_reliability = context["legacy_reliability"]
        product_base = _apply_empirical_bayes_draws(
            context["raw_product_draws"],
            pooled_sample,
            eb_weights[upc],
        )
        product_base["pooled_bootstrap_id"] = np.arange(
            config.bootstrap_replications
        )
        product_draws = _expand_persistence_draws(product_base, config)
        base_demand, regular_price, unit_cost = _product_economic_references(
            calibration,
            pooled_draws,
        )
        _annotate_product_draws(
            (product_base, product_draws),
            upc=upc,
            product_name=product_name,
            base_demand=base_demand,
            regular_price=regular_price,
            unit_cost=unit_cost,
            legacy_reliability=legacy_reliability,
        )
        product_draws["draw_weight"] /= product_draws["draw_weight"].sum()

        draw_frames.append(product_draws)
        base_frames.append(product_base)
        summary_rows.append(
            _product_summary_row(
                upc=upc,
                product_name=product_name,
                product_frame=product_frame,
                calibration=calibration,
                evaluation=evaluation,
                event_summary=event_summary,
                depth_events=depth_events,
                common_panels=common_panels,
                selected_gap=selected_gap,
                calibration_discount=calibration_discount,
                legacy_reliability=legacy_reliability,
                product_draws=product_draws,
                base_demand=base_demand,
                regular_price=regular_price,
                unit_cost=unit_cost,
            )
        )
        holdout_rows.extend(
            _product_holdout_rows(
                evaluation,
                product_draws,
                upc=upc,
                product_name=product_name,
                config=config,
            )
        )

    eb_detail["legacy_heuristic_reliability"] = eb_detail["upc"].map(
        {context["upc"]: context["legacy_reliability"] for context in contexts}
    )
    eb_detail["eb_minus_legacy_heuristic"] = (
        eb_detail["eb_weight"] - eb_detail["legacy_heuristic_reliability"]
    )

    product_holdout = pd.DataFrame(holdout_rows)
    return {
        "product_draws": pd.concat(draw_frames, ignore_index=True),
        "product_base_draws": pd.concat(base_frames, ignore_index=True),
        "product_summary": pd.DataFrame(summary_rows),
        "empirical_bayes_weights": eb_detail,
        "empirical_bayes_variance_summary": eb_summary,
        "product_event_study": (
            pd.concat(event_studies, ignore_index=True)
            if event_studies
            else pd.DataFrame()
        ),
        "product_holdout": product_holdout,
        "product_holdout_summary": _summarize_product_holdout(product_holdout),
    }

__all__ = ['CalibrationConfig', 'canonical_promotion_indicator', 'load_clean_history', 'calibrate_products']


# ---------------------------------------------------------------------------
# Pooled-calibration notebook helpers
# ---------------------------------------------------------------------------

POST_LAGS_FOR_CALIBRATION = (1, 2, 3, 4)
DISPLACEMENT_BOUNDS = (0.0, 3.0)
PERSISTENCE_BOUNDS = (0.05, 0.95)
MIN_PERSISTENCE_SIGNAL = 0.05
MAX_PERSISTENCE_SLOPE_RATIO = 1.25
PERSISTENCE_PARTIAL_ID_GRID = (0.15, 0.35, 0.60)

def numeric_series(
    frame: pd.DataFrame,
    candidates: list[str],
    default: float = np.nan,
) -> pd.Series:
    column = first_existing_column(
        frame,
        candidates,
        required=False,
    )
    if column is None:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")

def alternating_multiway_residual(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
    max_iterations: int = 300,
    tolerance: float = 1e-10,
) -> pd.Series:
    values = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    ).astype(float)

    if values.isna().any():
        raise ValueError(
            f"{value_column!r} contains missing values."
        )

    residual = values - values.mean()

    for _ in range(max_iterations):
        previous = residual.to_numpy(copy=True)

        for group_column in group_columns:
            residual = (
                residual
                - residual.groupby(
                    frame[group_column],
                    observed=True,
                ).transform("mean")
            )

        change = float(
            np.max(
                np.abs(
                    residual.to_numpy()
                    - previous
                )
            )
        )
        if change < tolerance:
            break

    return residual

def fit_persistence(
    row: pd.Series,
    *,
    post_lags: tuple[int, ...] = POST_LAGS_FOR_CALIBRATION,
    displacement_bounds: tuple[float, float] = DISPLACEMENT_BOUNDS,
    persistence_bounds: tuple[float, float] = PERSISTENCE_BOUNDS,
    minimum_signal: float = MIN_PERSISTENCE_SIGNAL,
    maximum_slope_ratio: float = MAX_PERSISTENCE_SLOPE_RATIO,
    partial_identification_grid: tuple[float, ...] = PERSISTENCE_PARTIAL_ID_GRID,
) -> tuple[float, bool, float]:
    """Map post-promotion depth slopes into persistence or a sensitivity value."""
    slopes = np.array(
        [row.get(f"post{lag}_depth_slope", np.nan) for lag in post_lags],
        dtype=float,
    )
    displacement = float(
        np.clip(
            slopes[0] if np.isfinite(slopes[0]) else 0.0,
            *displacement_bounds,
        )
    )
    second_slope = (
        float(slopes[1])
        if len(slopes) > 1 and np.isfinite(slopes[1])
        else np.nan
    )
    informative = bool(
        displacement >= minimum_signal
        and np.isfinite(second_slope)
        and second_slope > 0.0
        and second_slope <= maximum_slope_ratio * displacement
    )
    if not informative:
        return (
            float(np.median(partial_identification_grid)),
            False,
            np.nan,
        )

    persistence = float(
        np.clip(
            second_slope / displacement,
            *persistence_bounds,
        )
    )
    fit_loss = float(
        (second_slope - displacement * persistence) ** 2
    )
    return persistence, True, fit_loss


def bootstrap_behavioral_moments(
    event_summary_boot: pd.DataFrame,
    price_sufficient_boot: pd.DataFrame,
    common_panels: np.ndarray,
    regular_log_price_coefficient: float,
    *,
    bootstrap_replications: int,
    random_seed: int,
    elasticity_bounds: tuple[float, float],
    post_lags_for_calibration: tuple[int, ...],
) -> pd.DataFrame:
    """Cluster-bootstrap pooled elasticity, promotion lift, and displacement moments."""
    rng = np.random.default_rng(random_seed)
    bootstrap_rows: list[dict[str, float]] = []
    for bootstrap_id in range(
        bootstrap_replications
    ):
        sampled_panels = rng.choice(
            common_panels,
            size=len(common_panels),
            replace=True,
        )

        counts = (
            pd.Series(sampled_panels)
            .value_counts()
            .rename("cluster_weight")
        )

        events_boot = (
            event_summary_boot.merge(
                counts,
                left_on="store_upc",
                right_index=True,
                how="inner",
                validate="many_to_one",
            )
        )

        price_boot = (
            price_sufficient_boot.merge(
                counts,
                left_on="store_upc",
                right_index=True,
                how="inner",
                validate="one_to_one",
            )
        )

        event_weights = (
            events_boot[
                "cluster_weight"
            ].astype(float)
        )
        price_weights = (
            price_boot[
                "cluster_weight"
            ].astype(float)
        )

        elasticity_denominator = float(
            np.sum(
                price_weights
                * price_boot["xx"]
            )
        )
        elasticity_numerator = float(
            np.sum(
                price_weights
                * price_boot["xy"]
            )
        )

        log_price_coefficient = (
            elasticity_numerator
            / elasticity_denominator
            if elasticity_denominator > 1e-12
            else regular_log_price_coefficient
        )

        elasticity = float(
            np.clip(
                -log_price_coefficient,
                *elasticity_bounds,
            )
        )

        depth = events_boot[
            "event_discount_depth"
        ].clip(
            lower=0.0,
            upper=0.80,
        )

        ordinary_price_log_effect = (
            log_price_coefficient
            * events_boot[
                "current_price_change_residual"
            ]
        )

        promotion_lift_event = (
            events_boot[
                "current_lift_log"
            ]
            - ordinary_price_log_effect
        )

        row = {
            "bootstrap_id": bootstrap_id,
            "panels_sampled": (
                len(common_panels)
            ),
            "events_effective": float(
                event_weights.sum()
            ),
            "mean_discount_depth": (
                weighted_mean(
                    depth,
                    event_weights,
                )
            ),
            "mean_price_change_residual": (
                weighted_mean(
                    events_boot[
                        "current_price_change_residual"
                    ],
                    event_weights,
                )
            ),
            "current_lift_log": (
                weighted_mean(
                    events_boot[
                        "current_lift_log"
                    ],
                    event_weights,
                )
            ),
            "ordinary_price_lift_log": (
                weighted_mean(
                    ordinary_price_log_effect,
                    event_weights,
                )
            ),
            "promotion_lift_log": (
                weighted_mean(
                    promotion_lift_event,
                    event_weights,
                )
            ),
            "current_depth_slope": (
                weighted_slope(
                    depth,
                    events_boot[
                        "current_lift_log"
                    ],
                    event_weights,
                )
            ),
            "regular_log_price_coefficient": (
                log_price_coefficient
            ),
            "price_elasticity": elasticity,
        }

        for lag in (
            post_lags_for_calibration
        ):
            dip_column = (
                f"post{lag}_dip_log"
            )

            row[
                f"post{lag}_dip_log"
            ] = weighted_mean(
                events_boot[
                    dip_column
                ],
                event_weights,
            )

            row[
                f"post{lag}_depth_slope"
            ] = weighted_slope(
                depth,
                events_boot[
                    dip_column
                ],
                event_weights,
            )

        bootstrap_rows.append(row)
    return pd.DataFrame(bootstrap_rows)


def bootstrap_event_study_rows(
    event_plot_data: pd.DataFrame,
    descriptive_panels: np.ndarray,
    *,
    bootstrap_replications: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Cluster-bootstrap pooled and depth-specific event-study means."""
    rng_event = np.random.default_rng(random_seed)
    bootstrap_event_rows: list[dict[str, object]] = []
    bootstrap_depth_rows: list[dict[str, object]] = []
    for bootstrap_id in range(
        bootstrap_replications
    ):
        sampled_panels = rng_event.choice(
            descriptive_panels,
            size=len(descriptive_panels),
            replace=True,
        )

        weights = (
            pd.Series(sampled_panels)
            .value_counts()
            .rename("cluster_weight")
        )

        boot = event_plot_data.merge(
            weights,
            left_on="store_upc",
            right_index=True,
            how="inner",
            validate="many_to_one",
        )

        for relative_week, group in (
            boot.groupby(
                "relative_week",
                observed=True,
            )
        ):
            bootstrap_event_rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    "relative_week": int(
                        relative_week
                    ),
                    "mean_effect": (
                        weighted_mean(
                            group[
                                "event_effect"
                            ],
                            group[
                                "cluster_weight"
                            ].astype(float),
                        )
                    ),
                }
            )

        depth_boot = boot.loc[
            boot[
                "calibration_event"
            ]
        ]

        for (
            relative_week,
            depth_group,
        ), group in depth_boot.groupby(
            [
                "relative_week",
                "depth_group",
            ],
            observed=True,
        ):
            bootstrap_depth_rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    "relative_week": int(
                        relative_week
                    ),
                    "depth_group": (
                        str(depth_group)
                    ),
                    "mean_effect": (
                        weighted_mean(
                            group[
                                "event_effect"
                            ],
                            group[
                                "cluster_weight"
                            ].astype(float),
                        )
                    ),
                }
            )
    return bootstrap_event_rows, bootstrap_depth_rows



def load_panel_selection(
    history_raw: pd.DataFrame,
    prediction_path: Path,
    primary_model: str,
) -> tuple[set[str], int, int]:
    """Public wrapper for the panel and chronological-window selection."""
    return _load_panel_selection(
        history_raw=history_raw,
        prediction_path=prediction_path,
        primary_model=primary_model,
    )
