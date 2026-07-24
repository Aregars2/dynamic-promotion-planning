"""Behavioral calibration for product-level promotion dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PARAMETER_COLUMNS = [
    "price_elasticity",
    "promotion_lift_log",
    "displacement_strength",
    "inventory_persistence",
]

class CalibrationConfig:
    primary_model: str = "product_promotion"
    max_products: int = 8
    bootstrap_replications: int = 200
    random_seed: int = 42
    event_window: tuple[int, ...] = tuple(range(-4, 7))
    holdout_window: tuple[int, ...] = tuple(range(-3, 5))
    pre_event_weeks: tuple[int, ...] = (-3, -2, -1)
    post_lags: tuple[int, ...] = (1, 2, 3, 4)
    min_isolated_events: int = 15
    min_depth_events: int = 10
    min_common_panels: int = 4
    min_calibration_discount: float = 0.05
    fallback_calibration_discount: float = 0.02
    shrinkage_event_scale: float = 30.0
    shrinkage_panel_scale: float = 6.0
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

def load_clean_history(
    data_path: Path,
    prediction_path: Path,
    config: CalibrationConfig,
) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Demand data not found: {data_path}"
        )

    history_raw = pd.read_parquet(
        data_path
    )
    (
        selected_panels,
        calibration_end,
        evaluation_start,
    ) = _load_panel_selection(
        history_raw,
        prediction_path,
        config.primary_model,
    )

    history = history_raw.copy()

    for column in [
        "store",
        "upc",
        "store_upc",
    ]:
        history[column] = history[column].astype(str)

    history = history.loc[
        history["store_upc"].isin(
            selected_panels
        )
    ].copy()

    history["week"] = pd.to_numeric(
        history["week"],
        errors="coerce",
    )
    history["move"] = pd.to_numeric(
        history["move"],
        errors="coerce",
    )

    price_column = first_existing_column(
        history,
        [
            "model_unit_price",
            "unit_price_observed",
            "unit_price",
            "price",
        ],
    )
    history["price"] = pd.to_numeric(
        history[price_column],
        errors="coerce",
    )

    regular_price_column = first_existing_column(
        history,
        [
            "regular_price",
            "reference_price",
        ],
        required=False,
    )

    if regular_price_column is None:
        history["regular_price_model"] = (
            history.groupby(
                "store_upc",
                observed=True,
            )["price"]
            .transform(
                lambda values: values.rolling(
                    13,
                    min_periods=4,
                ).max()
            )
        )
    else:
        history["regular_price_model"] = pd.to_numeric(
            history[regular_price_column],
            errors="coerce",
        )

    history["regular_price_model"] = (
        history["regular_price_model"]
        .where(
            history["regular_price_model"].gt(0)
        )
        .fillna(
            history["price"]
        )
    )

    discount_column = first_existing_column(
        history,
        [
            "discount_depth_model",
            "discount_depth",
        ],
        required=False,
    )

    if discount_column is None:
        history["discount_depth_model"] = (
            1.0
            - history["price"]
            / history["regular_price_model"]
        )
    else:
        history["discount_depth_model"] = pd.to_numeric(
            history[discount_column],
            errors="coerce",
        )

    history["discount_depth_model"] = (
        history["discount_depth_model"]
        .fillna(0.0)
        .clip(
            0.0,
            0.80,
        )
    )

    promotion_column = first_existing_column(
        history,
        [
            "promotion_indicator",
            "promo_recorded",
            "promo_from_discount",
        ],
        required=False,
    )

    if promotion_column is None:
        promotion_indicator = (
            history["discount_depth_model"].ge(0.03)
        )
    else:
        promotion_indicator = (
            pd.to_numeric(
                history[promotion_column],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
        )

    history["promotion_indicator"] = (
        promotion_indicator
        | history["discount_depth_model"].ge(0.03)
    ).astype(int)

    post_column = first_existing_column(
        history,
        [
            "post_promotion_indicator",
            "post_promo",
        ],
        required=False,
    )

    if post_column is None:
        history["post_promotion_indicator"] = (
            history.groupby(
                "store_upc",
                observed=True,
            )["promotion_indicator"]
            .shift(1)
            .fillna(0)
            .astype(int)
        )
    else:
        history["post_promotion_indicator"] = (
            pd.to_numeric(
                history[post_column],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    price_imputed = (
        history["price_imputed"]
        .fillna(False)
        .astype(bool)
        if "price_imputed" in history.columns
        else pd.Series(
            False,
            index=history.index,
        )
    )

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

    history = history.loc[
        valid
    ].copy()
    history["week"] = history["week"].astype(int)

    if history.duplicated(
        [
            "store_upc",
            "week",
        ]
    ).any():
        raise ValueError(
            "Duplicate store-product-week rows found."
        )

    history["log1p_move"] = np.log1p(
        history["move"]
    )
    history["log_price"] = np.log(
        history["price"]
    )

    margin_column = (
        "gross_margin_pct_observed"
    )

    if margin_column not in history.columns:
        raise KeyError(
            f"Required margin column {margin_column!r} missing."
        )

    raw_margin = pd.to_numeric(
        history[margin_column],
        errors="coerce",
    )
    finite_margin = raw_margin.loc[
        raw_margin.notna()
        & np.isfinite(raw_margin)
    ]
    margin_scale = (
        100.0
        if (
            not finite_margin.empty
            and finite_margin.abs().quantile(
                0.95
            )
            > 1.5
        )
        else 1.0
    )
    history["gross_margin_fraction"] = (
        raw_margin / margin_scale
    )
    valid_margin = history[
        "gross_margin_fraction"
    ].between(
        -1.0,
        0.95,
        inclusive="both",
    )
    history["unit_cost_observed"] = (
        history["price"]
        * (
            1.0
            - history["gross_margin_fraction"]
        )
    )
    history.loc[
        ~valid_margin
        | ~np.isfinite(
            history["unit_cost_observed"]
        )
        | history["unit_cost_observed"].le(0),
        "unit_cost_observed",
    ] = np.nan

    history["sample_period"] = np.where(
        history["week"].le(
            calibration_end
        ),
        "calibration",
        "evaluation",
    )

    coverage = (
        history.groupby(
            "upc",
            observed=True,
        )
        .agg(
            panels=(
                "store_upc",
                "nunique",
            ),
            rows=(
                "week",
                "size",
            ),
            promotion_rows=(
                "promotion_indicator",
                "sum",
            ),
        )
        .reset_index()
    )
    coverage["selection_score"] = (
        coverage["panels"]
        * np.log1p(
            coverage["promotion_rows"]
        )
    )
    selected_products = (
        coverage.sort_values(
            [
                "selection_score",
                "rows",
            ],
            ascending=False,
        )
        .head(
            config.max_products
        )["upc"]
        .astype(str)
        .tolist()
    )

    history = history.loc[
        history["upc"].isin(
            selected_products
        )
    ].copy()

    description_column = first_existing_column(
        history,
        [
            "descrip",
            "description",
            "product_description",
        ],
        required=False,
    )

    if description_column is None:
        product_names = {
            upc: upc
            for upc in selected_products
        }
    else:
        product_names = (
            history.groupby(
                "upc",
                observed=True,
            )[description_column]
            .agg(
                lambda values: (
                    values.dropna()
                    .astype(str)
                    .iloc[0]
                    if not values.dropna().empty
                    else ""
                )
            )
            .to_dict()
        )

    history = (
        history.sort_values(
            [
                "upc",
                "store_upc",
                "week",
            ]
        )
        .reset_index(drop=True)
    )

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

def build_event_data(
    product_period: pd.DataFrame,
    event_window: tuple[int, ...],
    pre_event_weeks: tuple[int, ...],
    minimum_events: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[int, int],
]:
    selected_gap = (1, 1)
    selected_mask = pd.Series(
        False,
        index=product_period.index,
    )

    for gap in [
        (3, 3),
        (2, 2),
        (1, 1),
    ]:
        candidate = isolated_promotion_mask(
            product_period,
            pre_gap=gap[0],
            post_gap=gap[1],
        )
        selected_gap = gap
        selected_mask = candidate

        if int(candidate.sum()) >= minimum_events:
            break

    events = (
        product_period.loc[
            selected_mask,
            [
                "upc",
                "store",
                "store_upc",
                "week",
                "discount_depth_model",
            ],
        ]
        .copy()
        .rename(
            columns={
                "week": "event_week",
                "discount_depth_model": (
                    "event_discount_depth"
                ),
            }
        )
    )
    events["event_id"] = np.arange(
        len(events),
        dtype=int,
    )

    if events.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            selected_gap,
        )

    source = product_period[
        [
            "store_upc",
            "week",
            "demand_residual",
            "price_residual",
            "move",
        ]
    ].copy()
    blocks = []

    for relative_week in event_window:
        block = events[
            [
                "event_id",
                "upc",
                "store_upc",
                "event_week",
                "event_discount_depth",
            ]
        ].copy()
        block["relative_week"] = int(
            relative_week
        )
        block["week"] = (
            block["event_week"]
            + int(relative_week)
        )
        block = block.merge(
            source,
            on=[
                "store_upc",
                "week",
            ],
            how="left",
            validate="many_to_one",
        )
        blocks.append(block)

    event_long = pd.concat(
        blocks,
        ignore_index=True,
    )

    pre_means = (
        event_long.loc[
            event_long["relative_week"].isin(
                pre_event_weeks
            )
        ]
        .groupby(
            "event_id",
            observed=True,
        )
        .agg(
            pre_demand_mean=(
                "demand_residual",
                "mean",
            ),
            pre_price_mean=(
                "price_residual",
                "mean",
            ),
            pre_observations=(
                "demand_residual",
                "count",
            ),
        )
        .reset_index()
    )

    event_long = event_long.merge(
        pre_means,
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    event_long = event_long.loc[
        event_long["pre_observations"].ge(2)
    ].copy()
    event_long["event_effect"] = (
        event_long["demand_residual"]
        - event_long["pre_demand_mean"]
    )
    event_long["event_price_change"] = (
        event_long["price_residual"]
        - event_long["pre_price_mean"]
    )

    event_summary = (
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
            [
                "event_id",
                "event_price_change",
            ],
        ]
        .drop_duplicates(
            "event_id"
        )
        .rename(
            columns={
                "event_price_change": (
                    "current_price_change"
                )
            }
        )
    )

    event_summary = event_summary.merge(
        current_price,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    required = [
        column
        for column in [
            0,
            1,
            "current_price_change",
        ]
        if column in event_summary.columns
    ]
    event_summary = event_summary.dropna(
        subset=required
    ).copy()

    if 0 in event_summary.columns:
        event_summary = event_summary.rename(
            columns={
                0: "current_lift_log",
            }
        )
    else:
        event_summary["current_lift_log"] = np.nan

    for lag in [
        1,
        2,
        3,
        4,
    ]:
        event_summary[
            f"post{lag}_dip_log"
        ] = (
            -event_summary[lag]
            if lag in event_summary.columns
            else np.nan
        )

    return (
        event_long,
        event_summary,
        selected_gap,
    )

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

def calibrate_products(
    history: pd.DataFrame,
    selected_products: list[str],
    product_names: dict[str, str],
    pooled_draws: pd.DataFrame,
    config: CalibrationConfig,
) -> dict[str, pd.DataFrame]:
    pooled_valid = pooled_draws.dropna(
        subset=PARAMETER_COLUMNS
    ).copy()
    pooled_probabilities = (
        pooled_valid["draw_weight"].to_numpy(
            dtype=float
        )
        if "draw_weight" in pooled_valid.columns
        else np.ones(
            len(pooled_valid),
            dtype=float,
        )
    )
    pooled_probabilities = (
        pooled_probabilities
        / pooled_probabilities.sum()
    )
    rng_master = np.random.default_rng(
        config.random_seed
    )

    draw_frames = []
    base_frames = []
    summary_rows = []
    event_study_frames = []
    holdout_rows = []

    for product_index, upc in enumerate(
        selected_products
    ):
        product_frame = history.loc[
            history["upc"].eq(upc)
        ].copy()
        calibration = prepare_product_period(
            product_frame,
            "calibration",
        )
        evaluation = prepare_product_period(
            product_frame,
            "evaluation",
        )
        (
            event_long,
            event_summary,
            selected_gap,
        ) = build_event_data(
            calibration,
            config.event_window,
            config.pre_event_weeks,
            config.min_isolated_events,
        )

        if not event_long.empty:
            event_study = (
                event_long.groupby(
                    "relative_week",
                    observed=True,
                )
                .agg(
                    observed_effect=(
                        "event_effect",
                        "mean",
                    ),
                    events=(
                        "event_id",
                        "nunique",
                    ),
                )
                .reset_index()
            )
            event_study["upc"] = upc
            event_study["product_name"] = (
                product_names.get(
                    upc,
                    upc,
                )
            )
            event_study_frames.append(
                event_study
            )

        calibration_discount = (
            config.min_calibration_discount
        )
        depth_events = (
            event_summary.loc[
                event_summary[
                    "event_discount_depth"
                ].ge(
                    calibration_discount
                )
            ].copy()
            if not event_summary.empty
            else pd.DataFrame()
        )

        if len(depth_events) < config.min_depth_events:
            calibration_discount = (
                config.fallback_calibration_discount
            )
            depth_events = (
                event_summary.loc[
                    event_summary[
                        "event_discount_depth"
                    ].ge(
                        calibration_discount
                    )
                ].copy()
                if not event_summary.empty
                else pd.DataFrame()
            )

        regular_rows = calibration.loc[
            calibration["promotion_indicator"].eq(0)
            & calibration[
                "post_promotion_indicator"
            ].eq(0)
        ].copy()
        panel_price = (
            regular_rows.assign(
                xx=lambda frame: (
                    frame["price_residual"] ** 2
                ),
                xy=lambda frame: (
                    frame["price_residual"]
                    * frame["demand_residual"]
                ),
            )
            .groupby(
                "store_upc",
                observed=True,
            )
            .agg(
                xx=(
                    "xx",
                    "sum",
                ),
                xy=(
                    "xy",
                    "sum",
                ),
                observations=(
                    "week",
                    "size",
                ),
            )
            .reset_index()
        )

        event_panels = set(
            depth_events["store_upc"].unique()
            if not depth_events.empty
            else []
        )
        price_panels = set(
            panel_price["store_upc"].unique()
        )
        common_panels = np.array(
            sorted(
                event_panels.intersection(
                    price_panels
                )
            )
        )

        bootstrap = _bootstrap_product(
            depth_events,
            panel_price,
            common_panels,
            config,
            config.random_seed
            + product_index
            + 1,
        )
        pooled_indices = rng_master.choice(
            len(pooled_valid),
            size=config.bootstrap_replications,
            replace=True,
            p=pooled_probabilities,
        )
        pooled_sample = (
            pooled_valid.iloc[
                pooled_indices
            ][PARAMETER_COLUMNS]
            .reset_index(drop=True)
        )

        event_reliability = (
            len(depth_events)
            / (
                len(depth_events)
                + config.shrinkage_event_scale
            )
            if len(depth_events) > 0
            else 0.0
        )
        panel_reliability = (
            len(common_panels)
            / (
                len(common_panels)
                + config.shrinkage_panel_scale
            )
            if len(common_panels) > 0
            else 0.0
        )
        reliability = float(
            np.clip(
                event_reliability
                * panel_reliability,
                0.0,
                1.0,
            )
        )

        if bootstrap.empty:
            product_base = (
                pooled_sample.copy()
            )
            product_base[
                "bootstrap_id"
            ] = np.arange(
                config.bootstrap_replications
            )
            product_base[
                "post1_depth_slope"
            ] = product_base[
                "displacement_strength"
            ]
            product_base[
                "post2_depth_slope"
            ] = (
                product_base[
                    "displacement_strength"
                ]
                * product_base[
                    "inventory_persistence"
                ]
            )
        else:
            product_base = bootstrap.copy()

            for parameter in [
                "price_elasticity",
                "promotion_lift_log",
            ]:
                product_base[parameter] = (
                    reliability
                    * product_base[parameter]
                    + (
                        1.0 - reliability
                    )
                    * pooled_sample[parameter]
                )

            product_displacement = (
                product_base[
                    "post1_depth_slope"
                ]
                .fillna(
                    pooled_sample[
                        "displacement_strength"
                    ]
                )
                .clip(
                    0.0,
                    3.0,
                )
            )
            product_base[
                "displacement_strength"
            ] = (
                reliability
                * product_displacement
                + (
                    1.0 - reliability
                )
                * pooled_sample[
                    "displacement_strength"
                ]
            )
            product_base[
                "inventory_persistence"
            ] = pooled_sample[
                "inventory_persistence"
            ]

        product_base[
            "price_elasticity"
        ] = product_base[
            "price_elasticity"
        ].clip(
            0.10,
            5.00,
        )
        product_base[
            "promotion_lift_log"
        ] = product_base[
            "promotion_lift_log"
        ].clip(
            -1.00,
            3.00,
        )
        product_base[
            "displacement_strength"
        ] = product_base[
            "displacement_strength"
        ].clip(
            0.00,
            3.00,
        )

        expanded_rows = []

        for row in product_base.itertuples(
            index=False
        ):
            row_dict = row._asdict()
            first_slope = row_dict.get(
                "post1_depth_slope",
                np.nan,
            )
            second_slope = row_dict.get(
                "post2_depth_slope",
                np.nan,
            )
            informative = bool(
                np.isfinite(first_slope)
                and np.isfinite(second_slope)
                and first_slope
                >= config.min_persistence_signal
                and second_slope > 0
                and second_slope
                <= (
                    config.max_persistence_slope_ratio
                    * first_slope
                )
            )

            if informative:
                persistence_values = [
                    float(
                        np.clip(
                            second_slope
                            / first_slope,
                            0.05,
                            0.95,
                        )
                    )
                ]
                persistence_source = (
                    "product_depth_decay"
                )
            else:
                pooled_persistence = float(
                    np.clip(
                        row_dict.get(
                            "inventory_persistence",
                            np.median(
                                config.persistence_grid
                            ),
                        ),
                        0.05,
                        0.95,
                    )
                )
                persistence_values = sorted(
                    set(
                        [
                            *config.persistence_grid,
                            pooled_persistence,
                        ]
                    )
                )
                persistence_source = (
                    "partial_id_plus_pooled"
                )

            for persistence in persistence_values:
                expanded = dict(row_dict)
                expanded[
                    "inventory_persistence"
                ] = float(persistence)
                expanded[
                    "persistence_source"
                ] = persistence_source
                expanded[
                    "persistence_informative"
                ] = informative
                expanded[
                    "draw_weight"
                ] = (
                    1.0
                    / (
                        config.bootstrap_replications
                        * len(
                            persistence_values
                        )
                    )
                )
                expanded_rows.append(expanded)

        product_draws = pd.DataFrame(
            expanded_rows
        )

        regular_reference = calibration.loc[
            calibration["promotion_indicator"].eq(0)
            & calibration[
                "post_promotion_indicator"
            ].eq(0)
        ].copy()
        base_demand = float(
            regular_reference["move"].median()
        )
        regular_price = float(
            regular_reference[
                "regular_price_model"
            ].median()
        )
        unit_cost = float(
            calibration[
                "unit_cost_observed"
            ].median()
        )

        if (
            not np.isfinite(base_demand)
            or base_demand <= 0
        ):
            base_demand = float(
                calibration["move"].median()
            )

        if (
            not np.isfinite(regular_price)
            or regular_price <= 0
        ):
            regular_price = float(
                calibration["price"].median()
            )

        if (
            not np.isfinite(unit_cost)
            or unit_cost <= 0
        ):
            unit_cost = float(
                pooled_draws[
                    "unit_cost"
                ].median()
            )

        for frame in [
            product_base,
            product_draws,
        ]:
            frame["upc"] = upc
            frame["product_name"] = (
                product_names.get(
                    upc,
                    upc,
                )
            )
            frame["base_demand"] = base_demand
            frame["regular_price"] = regular_price
            frame["unit_cost"] = unit_cost
            frame[
                "shrinkage_reliability"
            ] = reliability

        product_draws["draw_weight"] = (
            product_draws["draw_weight"]
            / product_draws[
                "draw_weight"
            ].sum()
        )

        draw_frames.append(product_draws)
        base_frames.append(product_base)
        summary_rows.append(
            {
                "upc": upc,
                "product_name": (
                    product_names.get(
                        upc,
                        upc,
                    )
                ),
                "calibration_rows": int(
                    len(calibration)
                ),
                "evaluation_rows": int(
                    len(evaluation)
                ),
                "panels": int(
                    product_frame[
                        "store_upc"
                    ].nunique()
                ),
                "isolated_events": int(
                    len(event_summary)
                ),
                "depth_events": int(
                    len(depth_events)
                ),
                "common_panels": int(
                    len(common_panels)
                ),
                "selected_gap": (
                    f"{selected_gap[0]}/"
                    f"{selected_gap[1]}"
                ),
                "calibration_discount": (
                    calibration_discount
                ),
                "shrinkage_reliability": (
                    reliability
                ),
                "price_elasticity_median": float(
                    product_draws[
                        "price_elasticity"
                    ].median()
                ),
                "promotion_lift_median": float(
                    product_draws[
                        "promotion_lift_log"
                    ].median()
                ),
                "displacement_median": float(
                    product_draws[
                        "displacement_strength"
                    ].median()
                ),
                "persistence_median": float(
                    product_draws[
                        "inventory_persistence"
                    ].median()
                ),
                "persistence_informative_share": float(
                    product_draws[
                        "persistence_informative"
                    ].mean()
                ),
                "base_demand": base_demand,
                "regular_price": regular_price,
                "unit_cost": unit_cost,
            }
        )

        (
            holdout_long,
            holdout_summary,
            holdout_gap,
        ) = build_event_data(
            evaluation,
            config.holdout_window,
            config.pre_event_weeks,
            minimum_events=5,
        )

        if not holdout_summary.empty:
            theta_draws = product_draws[
                PARAMETER_COLUMNS
            ].to_numpy(
                dtype=float
            )
            draw_weights = product_draws[
                "draw_weight"
            ].to_numpy(
                dtype=float
            )
            depths = holdout_summary[
                "event_discount_depth"
            ].to_numpy(
                dtype=float
            )

            for relative_week in (
                config.holdout_window
            ):
                observed = holdout_long.loc[
                    holdout_long[
                        "relative_week"
                    ].eq(
                        relative_week
                    ),
                    "event_effect",
                ]

                if relative_week < 0:
                    prediction_matrix = np.zeros(
                        (
                            len(depths),
                            len(theta_draws),
                        ),
                        dtype=float,
                    )
                elif relative_week == 0:
                    current_price_changes = (
                        holdout_summary[
                            "current_price_change"
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    prediction_matrix = (
                        -theta_draws[
                            None,
                            :,
                            0,
                        ]
                        * current_price_changes[
                            :,
                            None,
                        ]
                        + theta_draws[
                            None,
                            :,
                            1,
                        ]
                    )
                else:
                    prediction_matrix = (
                        -depths[:, None]
                        * theta_draws[
                            None,
                            :,
                            2,
                        ]
                        * np.power(
                            theta_draws[
                                None,
                                :,
                                3,
                            ],
                            relative_week - 1,
                        )
                    )

                draw_predictions = (
                    prediction_matrix.mean(
                        axis=0
                    )
                )
                predicted_mean = float(
                    np.average(
                        draw_predictions,
                        weights=draw_weights,
                    )
                )
                observed_mean = float(
                    observed.mean()
                )
                holdout_rows.append(
                    {
                        "upc": upc,
                        "product_name": (
                            product_names.get(
                                upc,
                                upc,
                            )
                        ),
                        "relative_week": int(
                            relative_week
                        ),
                        "observed_effect": (
                            observed_mean
                        ),
                        "predicted_effect": (
                            predicted_mean
                        ),
                        "predicted_q10": (
                            weighted_quantile(
                                draw_predictions,
                                draw_weights,
                                0.10,
                            )
                        ),
                        "predicted_q90": (
                            weighted_quantile(
                                draw_predictions,
                                draw_weights,
                                0.90,
                            )
                        ),
                        "absolute_error": float(
                            abs(
                                observed_mean
                                - predicted_mean
                            )
                        ),
                        "events": int(
                            holdout_summary[
                                "event_id"
                            ].nunique()
                        ),
                        "selected_gap": (
                            f"{holdout_gap[0]}/"
                            f"{holdout_gap[1]}"
                        ),
                    }
                )

    product_draws_all = pd.concat(
        draw_frames,
        ignore_index=True,
    )
    product_base_all = pd.concat(
        base_frames,
        ignore_index=True,
    )
    product_summary = pd.DataFrame(
        summary_rows
    )
    product_event_study = (
        pd.concat(
            event_study_frames,
            ignore_index=True,
        )
        if event_study_frames
        else pd.DataFrame()
    )
    product_holdout = pd.DataFrame(
        holdout_rows
    )

    if not product_holdout.empty:
        product_holdout_summary = (
            product_holdout.groupby(
                [
                    "upc",
                    "product_name",
                ],
                observed=True,
            )
            .agg(
                holdout_events=(
                    "events",
                    "max",
                ),
                mean_absolute_error=(
                    "absolute_error",
                    "mean",
                ),
                post_period_mae=(
                    "absolute_error",
                    lambda values: float(
                        values.iloc[4:].mean()
                    ),
                ),
            )
            .reset_index()
        )
    else:
        product_holdout_summary = (
            pd.DataFrame()
        )

    return {
        "product_draws": (
            product_draws_all
        ),
        "product_base_draws": (
            product_base_all
        ),
        "product_summary": (
            product_summary
        ),
        "product_event_study": (
            product_event_study
        ),
        "product_holdout": (
            product_holdout
        ),
        "product_holdout_summary": (
            product_holdout_summary
        ),
    }

__all__ = ['CalibrationConfig', 'load_clean_history', 'calibrate_products']
