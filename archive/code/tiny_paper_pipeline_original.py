
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    linear_sum_assignment,
    milp,
    minimize_scalar,
)
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


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


@dataclass(frozen=True)
class PlanningConfig:
    horizon: int = 12
    washout_horizon: int = 6
    actions: tuple[float, ...] = (0.00, 0.10, 0.20, 0.30)
    cooldown: int = 2
    max_promotions: int = 4
    discount_factor: float = 0.995
    alpha_min: float = 0.0
    alpha_max: float = 4.0
    alpha_step: float = 0.01
    support_count_threshold: int = 20


@dataclass(frozen=True)
class ProductEconomics:
    base_demand: float
    regular_price: float
    unit_cost: float


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    candidates = [current, current.parent]

    root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "data" / "processed").is_dir()
        ),
        None,
    )

    if root is None:
        raise FileNotFoundError(
            f"Could not find project root from {current}"
        )

    return root


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


def build_action_support(
    history: pd.DataFrame,
    selected_products: list[str],
    product_names: dict[str, str],
    actions: tuple[float, ...],
    bandwidth: float,
    threshold: int,
) -> pd.DataFrame:
    calibration = history.loc[
        history["sample_period"].eq(
            "calibration"
        )
    ].copy()
    rows = []

    for upc in selected_products:
        product_rows = calibration.loc[
            calibration["upc"].eq(upc)
        ]

        for action in actions:
            if action <= 1e-12:
                matched = product_rows.loc[
                    product_rows[
                        "promotion_indicator"
                    ].eq(0)
                ]
            else:
                matched = product_rows.loc[
                    product_rows[
                        "promotion_indicator"
                    ].eq(1)
                    & product_rows[
                        "discount_depth_model"
                    ].sub(
                        action
                    ).abs().le(
                        bandwidth
                    )
                ]

            rows.append(
                {
                    "upc": upc,
                    "product_name": (
                        product_names.get(
                            upc,
                            upc,
                        )
                    ),
                    "discount_depth": float(
                        action
                    ),
                    "support_count": int(
                        len(matched)
                    ),
                    "support_panels": int(
                        matched[
                            "store_upc"
                        ].nunique()
                    ),
                    "well_supported": bool(
                        len(matched)
                        >= threshold
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_observed_calendar(
    history: pd.DataFrame,
    selected_products: list[str],
    product_names: dict[str, str],
    actions: tuple[float, ...],
    horizon: int,
) -> pd.DataFrame:
    evaluation = history.loc[
        history["sample_period"].eq(
            "evaluation"
        )
    ].copy()
    evaluation_weeks = sorted(
        evaluation["week"].unique()
    )[:horizon]

    if not evaluation_weeks:
        return pd.DataFrame()

    output = (
        evaluation.loc[
            evaluation["week"].isin(
                evaluation_weeks
            )
        ]
        .groupby(
            [
                "upc",
                "week",
            ],
            observed=True,
        )
        .agg(
            stores=(
                "store",
                "nunique",
            ),
            promotion_store_share=(
                "promotion_indicator",
                "mean",
            ),
            median_promotion_depth=(
                "discount_depth_model",
                lambda values: float(
                    values.loc[
                        values.gt(0.02)
                    ].median()
                    if values.gt(0.02).any()
                    else 0.0
                ),
            ),
        )
        .reset_index()
    )
    output["product_name"] = output[
        "upc"
    ].map(product_names)
    output["promoted"] = output[
        "promotion_store_share"
    ].ge(0.25)
    action_array = np.asarray(
        actions,
        dtype=float,
    )
    output["observed_action"] = 0.0
    promoted = output["promoted"]
    output.loc[
        promoted,
        "observed_action",
    ] = [
        float(
            action_array[
                np.argmin(
                    np.abs(
                        action_array - depth
                    )
                )
            ]
        )
        for depth in output.loc[
            promoted,
            "median_promotion_depth",
        ]
    ]
    output["relative_week"] = (
        output["week"]
        - min(evaluation_weeks)
        + 1
    )

    complete_index = pd.MultiIndex.from_product(
        [
            selected_products,
            range(
                1,
                horizon + 1,
            ),
        ],
        names=[
            "upc",
            "relative_week",
        ],
    )
    output = (
        output.set_index(
            [
                "upc",
                "relative_week",
            ]
        )
        .reindex(
            complete_index
        )
        .reset_index()
    )
    output["product_name"] = output[
        "upc"
    ].map(product_names)
    output[
        "promotion_store_share"
    ] = output[
        "promotion_store_share"
    ].fillna(0.0)
    output[
        "median_promotion_depth"
    ] = output[
        "median_promotion_depth"
    ].fillna(0.0)
    output["promoted"] = output[
        "promoted"
    ].fillna(False)
    output["observed_action"] = output[
        "observed_action"
    ].fillna(0.0)

    return output


def enumerate_schedules(
    config: PlanningConfig,
) -> np.ndarray:
    actions = np.asarray(
        config.actions,
        dtype=float,
    )
    schedules = []

    def recurse(
        time_index: int,
        cooldown_remaining: int,
        promotion_count: int,
        path: list[float],
    ) -> None:
        if time_index == config.horizon:
            schedules.append(
                tuple(path)
            )
            return

        if cooldown_remaining > 0:
            feasible_actions = [
                0.0
            ]
        else:
            feasible_actions = [
                float(action)
                for action in actions
                if (
                    action <= 1e-12
                    or promotion_count
                    < config.max_promotions
                )
            ]

        for depth in feasible_actions:
            promoted = depth > 1e-12
            path.append(depth)
            recurse(
                time_index + 1,
                (
                    config.cooldown
                    if promoted
                    else max(
                        cooldown_remaining - 1,
                        0,
                    )
                ),
                promotion_count
                + int(promoted),
                path,
            )
            path.pop()

    recurse(
        0,
        0,
        0,
        [],
    )

    return np.asarray(
        schedules,
        dtype=float,
    )


def compute_value_matrices(
    schedules: np.ndarray,
    theta: np.ndarray,
    economics: ProductEconomics,
    discount_factor: float,
    washout_horizon: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    schedule_count = len(schedules)
    draw_count = len(theta)
    base_matrix = np.empty(
        (
            schedule_count,
            draw_count,
        ),
        dtype=np.float32,
    )
    exposure_matrix = np.empty_like(
        base_matrix
    )

    for draw_index, draw in enumerate(
        theta
    ):
        (
            elasticity,
            promotion_lift,
            displacement,
            persistence,
        ) = draw
        inventory = np.zeros(
            schedule_count,
            dtype=float,
        )
        base_value = np.zeros(
            schedule_count,
            dtype=float,
        )
        exposure = np.zeros(
            schedule_count,
            dtype=float,
        )

        for time_index in range(
            schedules.shape[1]
        ):
            depth = schedules[
                :,
                time_index,
            ]
            demand = (
                economics.base_demand
                * np.power(
                    np.maximum(
                        1.0 - depth,
                        1e-8,
                    ),
                    -elasticity,
                )
                * np.exp(
                    promotion_lift
                    * (
                        depth > 1e-12
                    )
                    - displacement
                    * np.maximum(
                        inventory,
                        0.0,
                    )
                )
            )
            price = (
                economics.regular_price
                * (
                    1.0 - depth
                )
            )
            discount_weight = (
                discount_factor
                ** time_index
            )
            base_value += (
                discount_weight
                * (
                    price
                    - economics.unit_cost
                )
                * demand
            )
            exposure += (
                discount_weight
                * depth
                * demand
            )
            inventory = (
                persistence
                * np.maximum(
                    inventory,
                    0.0,
                )
                + depth
            )

        if washout_horizon < 0:
            raise ValueError(
                "washout_horizon must be nonnegative."
            )

        for washout_index in range(
            washout_horizon
        ):
            time_index = (
                schedules.shape[
                    1
                ]
                + washout_index
            )
            demand = (
                economics.base_demand
                * np.exp(
                    -displacement
                    * np.maximum(
                        inventory,
                        0.0,
                    )
                )
            )
            discount_weight = (
                discount_factor
                ** time_index
            )
            base_value += (
                discount_weight
                * (
                    economics.regular_price
                    - economics.unit_cost
                )
                * demand
            )
            inventory = (
                persistence
                * np.maximum(
                    inventory,
                    0.0,
                )
            )

        base_matrix[
            :,
            draw_index,
        ] = base_value
        exposure_matrix[
            :,
            draw_index,
        ] = exposure

    return (
        base_matrix,
        exposure_matrix,
    )


def _pareto_within_indices(
    indices: np.ndarray,
    base_values: np.ndarray,
    exposures: np.ndarray,
    tolerance: float = 1e-9,
) -> np.ndarray:
    ordered = sorted(
        [
            int(index)
            for index in indices
        ],
        key=lambda index: (
            -float(
                exposures[index]
            ),
            -float(
                base_values[index]
            ),
            index,
        ),
    )
    retained = []
    best_base = -np.inf

    for index in ordered:
        base = float(
            base_values[index]
        )

        if base > best_base + tolerance:
            retained.append(index)
            best_base = base

    return np.asarray(
        retained,
        dtype=int,
    )


def _schedule_support(
    upc: str,
    schedule: np.ndarray,
    action_support: pd.DataFrame,
    threshold: int,
) -> tuple[int, bool]:
    promoted_depths = np.unique(
        schedule[
            schedule > 1e-12
        ]
    )

    if len(promoted_depths) == 0:
        return (
            int(
                action_support.loc[
                    action_support["upc"].astype(
                        str
                    ).eq(upc)
                    & np.isclose(
                        action_support[
                            "discount_depth"
                        ],
                        0.0,
                    ),
                    "support_count",
                ].max()
            ),
            True,
        )

    counts = []

    for depth in promoted_depths:
        matched = action_support.loc[
            action_support["upc"].astype(
                str
            ).eq(upc)
            & np.isclose(
                action_support[
                    "discount_depth"
                ],
                depth,
            ),
            "support_count",
        ]
        counts.append(
            int(matched.iloc[0])
            if not matched.empty
            else 0
        )

    minimum = min(counts)

    return (
        minimum,
        bool(
            minimum >= threshold
        ),
    )


def _best_index(
    values: np.ndarray,
    promotion_count: np.ndarray,
    total_depth: np.ndarray,
    tolerance: float = 1e-9,
) -> int:
    maximum = float(values.max())
    candidates = np.flatnonzero(
        values
        >= maximum - tolerance
    )

    return int(
        min(
            candidates,
            key=lambda index: (
                int(
                    promotion_count[index]
                ),
                float(
                    total_depth[index]
                ),
                int(index),
            ),
        )
    )


def _myopic_product_schedule(
    theta: np.ndarray,
    weights: np.ndarray,
    economics: ProductEconomics,
    alpha: float,
    config: PlanningConfig,
) -> np.ndarray:
    (
        elasticity,
        promotion_lift,
        displacement,
        persistence,
    ) = np.average(
        theta,
        axis=0,
        weights=weights,
    )
    actions = np.asarray(
        config.actions,
        dtype=float,
    )
    inventory = 0.0
    cooldown_remaining = 0
    promotion_count = 0
    schedule = []

    for _ in range(config.horizon):
        if cooldown_remaining > 0:
            feasible_actions = np.array(
                [0.0]
            )
        else:
            feasible_actions = np.array(
                [
                    action
                    for action in actions
                    if (
                        action <= 1e-12
                        or promotion_count
                        < config.max_promotions
                    )
                ]
            )

        action_values = []

        for depth in feasible_actions:
            demand = (
                economics.base_demand
                * np.power(
                    max(
                        1.0 - depth,
                        1e-8,
                    ),
                    -elasticity,
                )
                * np.exp(
                    promotion_lift
                    * float(
                        depth > 1e-12
                    )
                    - displacement
                    * max(
                        inventory,
                        0.0,
                    )
                )
            )
            price = (
                economics.regular_price
                * (
                    1.0 - depth
                )
            )
            profit = (
                (
                    price
                    - economics.unit_cost
                    + alpha
                    * depth
                )
                * demand
            )
            action_values.append(
                (
                    float(profit),
                    float(depth),
                )
            )

        _, selected_depth = max(
            action_values,
            key=lambda item: (
                item[0],
                -item[1],
            ),
        )
        schedule.append(
            selected_depth
        )
        inventory = (
            persistence
            * max(
                inventory,
                0.0,
            )
            + selected_depth
        )

        if selected_depth > 1e-12:
            promotion_count += 1
            cooldown_remaining = (
                config.cooldown
            )
        else:
            cooldown_remaining = max(
                cooldown_remaining - 1,
                0,
            )

    return np.asarray(
        schedule,
        dtype=float,
    )


def build_schedule_frontiers(
    product_draws: pd.DataFrame,
    product_summary: pd.DataFrame,
    action_support: pd.DataFrame,
    config: PlanningConfig,
) -> dict[str, Any]:
    product_draws = product_draws.copy()
    product_draws["upc"] = (
        product_draws["upc"].astype(str)
    )
    product_summary = product_summary.copy()
    product_summary["upc"] = (
        product_summary["upc"].astype(str)
    )
    action_support = action_support.copy()
    action_support["upc"] = (
        action_support["upc"].astype(str)
    )

    products = product_summary[
        "upc"
    ].tolist()
    product_names = product_summary.set_index(
        "upc"
    )["product_name"].astype(str).to_dict()
    schedules = enumerate_schedules(
        config
    )
    promotion_count = (
        schedules > 1e-12
    ).sum(axis=1)
    total_depth = schedules.sum(axis=1)
    occupancy = (
        schedules > 1e-12
    ).astype(int)
    schedule_lookup = {
        tuple(schedule.tolist()): int(
            schedule_index
        )
        for schedule_index, schedule in enumerate(
            schedules
        )
    }
    no_promotion_index = int(
        np.flatnonzero(
            promotion_count == 0
        )[0]
    )
    alpha_grid = np.round(
        np.arange(
            config.alpha_min,
            config.alpha_max
            + 0.5
            * config.alpha_step,
            config.alpha_step,
        ),
        10,
    )

    occupancy_groups: dict[
        tuple[int, ...],
        list[int],
    ] = {}

    for schedule_index, row in enumerate(
        occupancy
    ):
        key = tuple(
            row.tolist()
        )
        occupancy_groups.setdefault(
            key,
            [],
        ).append(schedule_index)

    product_artifacts = {}
    candidate_indices_by_product = {}
    frontier_rows = []
    escalation_rows = []
    policy_rows = []
    vdo_rows = []

    for upc in products:
        draws = product_draws.loc[
            product_draws["upc"].eq(
                upc
            )
        ].copy()
        weights = draws[
            "draw_weight"
        ].to_numpy(
            dtype=float
        )
        weights = weights / weights.sum()
        theta = draws[
            PARAMETER_COLUMNS
        ].to_numpy(
            dtype=float
        )
        economics = ProductEconomics(
            base_demand=float(
                draws["base_demand"].median()
            ),
            regular_price=float(
                draws["regular_price"].median()
            ),
            unit_cost=float(
                draws["unit_cost"].median()
            ),
        )
        (
            base_matrix,
            exposure_matrix,
        ) = compute_value_matrices(
            schedules,
            theta,
            economics,
            config.discount_factor,
            washout_horizon=(
                config.washout_horizon
            ),
        )
        nominal_base = (
            base_matrix.astype(float)
            @ weights
        )
        nominal_exposure = (
            exposure_matrix.astype(float)
            @ weights
        )

        product_artifacts[upc] = {
            "product_name": (
                product_names[upc]
            ),
            "theta": theta,
            "weights": weights,
            "economics": economics,
            "base_matrix": base_matrix,
            "exposure_matrix": (
                exposure_matrix
            ),
            "nominal_base": (
                nominal_base
            ),
            "nominal_exposure": (
                nominal_exposure
            ),
        }

        retained_indices = []

        for indices in occupancy_groups.values():
            retained_indices.extend(
                _pareto_within_indices(
                    np.asarray(
                        indices,
                        dtype=int,
                    ),
                    nominal_base,
                    nominal_exposure,
                ).tolist()
            )

        retained_indices = np.asarray(
            sorted(
                set(
                    retained_indices
                )
            ),
            dtype=int,
        )
        candidate_indices_by_product[
            upc
        ] = retained_indices

        for schedule_index in retained_indices:
            (
                minimum_support,
                well_supported,
            ) = _schedule_support(
                upc,
                schedules[
                    schedule_index
                ],
                action_support,
                config.support_count_threshold,
            )
            row = {
                "upc": upc,
                "product_name": (
                    product_names[upc]
                ),
                "schedule_id": int(
                    schedule_index
                ),
                "nominal_base": float(
                    nominal_base[
                        schedule_index
                    ]
                ),
                "nominal_exposure": float(
                    nominal_exposure[
                        schedule_index
                    ]
                ),
                "promotion_count": int(
                    promotion_count[
                        schedule_index
                    ]
                ),
                "total_depth": float(
                    total_depth[
                        schedule_index
                    ]
                ),
                "minimum_action_support": int(
                    minimum_support
                ),
                "well_supported": bool(
                    well_supported
                ),
                "occupancy_pattern": "".join(
                    str(int(value))
                    for value in occupancy[
                        schedule_index
                    ]
                ),
            }

            for week_index, depth in enumerate(
                schedules[
                    schedule_index
                ],
                start=1,
            ):
                row[
                    f"week_{week_index}"
                ] = float(depth)

            frontier_rows.append(row)

        dynamic_indices = []
        myopic_indices = []

        for alpha in alpha_grid:
            values = (
                nominal_base
                + float(alpha)
                * nominal_exposure
            )
            dynamic_index = _best_index(
                values,
                promotion_count,
                total_depth,
            )
            myopic_path = _myopic_product_schedule(
                theta,
                weights,
                economics,
                float(alpha),
                config,
            )
            myopic_index = schedule_lookup[
                tuple(
                    myopic_path.tolist()
                )
            ]
            dynamic_indices.append(
                dynamic_index
            )
            myopic_indices.append(
                myopic_index
            )
            policy_rows.append(
                {
                    "upc": upc,
                    "product_name": (
                        product_names[upc]
                    ),
                    "alpha": float(alpha),
                    "dynamic_schedule_id": int(
                        dynamic_index
                    ),
                    "myopic_schedule_id": int(
                        myopic_index
                    ),
                    "dynamic_promotion_count": int(
                        promotion_count[
                            dynamic_index
                        ]
                    ),
                    "myopic_promotion_count": int(
                        promotion_count[
                            myopic_index
                        ]
                    ),
                    "dynamic_value": float(
                        values[dynamic_index]
                    ),
                    "myopic_full_horizon_value": float(
                        values[myopic_index]
                    ),
                    "value_of_dynamic_optimization": float(
                        values[dynamic_index]
                        - values[myopic_index]
                    ),
                }
            )

        dynamic_counts = promotion_count[
            np.asarray(
                dynamic_indices,
                dtype=int,
            )
        ]
        escalation = {
            "upc": upc,
            "product_name": (
                product_names[upc]
            ),
        }

        for target in [
            1,
            2,
            4,
        ]:
            reached = np.flatnonzero(
                dynamic_counts >= target
            )
            escalation[
                f"alpha_{target}_promotions"
            ] = (
                float(
                    alpha_grid[
                        reached[0]
                    ]
                )
                if len(reached) > 0
                else np.nan
            )

        escalation_rows.append(
            escalation
        )

        selected_alpha = 2.24
        position = int(
            np.argmin(
                np.abs(
                    alpha_grid
                    - selected_alpha
                )
            )
        )
        dynamic_index = dynamic_indices[
            position
        ]
        myopic_index = myopic_indices[
            position
        ]
        values = (
            nominal_base
            + selected_alpha
            * nominal_exposure
        )
        vdo_rows.append(
            {
                "upc": upc,
                "product_name": (
                    product_names[upc]
                ),
                "alpha": selected_alpha,
                "dynamic_schedule_id": int(
                    dynamic_index
                ),
                "myopic_schedule_id": int(
                    myopic_index
                ),
                "dynamic_value": float(
                    values[dynamic_index]
                ),
                "myopic_value": float(
                    values[myopic_index]
                ),
                "value_of_dynamic_optimization": float(
                    values[dynamic_index]
                    - values[myopic_index]
                ),
                "dynamic_promotion_count": int(
                    promotion_count[
                        dynamic_index
                    ]
                ),
                "myopic_promotion_count": int(
                    promotion_count[
                        myopic_index
                    ]
                ),
            }
        )

    return {
        "products": products,
        "product_names": product_names,
        "schedules": schedules,
        "promotion_count": (
            promotion_count
        ),
        "total_depth": total_depth,
        "occupancy": occupancy,
        "schedule_lookup": (
            schedule_lookup
        ),
        "no_promotion_index": (
            no_promotion_index
        ),
        "product_artifacts": (
            product_artifacts
        ),
        "candidate_indices_by_product": (
            candidate_indices_by_product
        ),
        "frontier_table": pd.DataFrame(
            frontier_rows
        ),
        "escalation_table": pd.DataFrame(
            escalation_rows
        ),
        "policy_grid": pd.DataFrame(
            policy_rows
        ),
        "product_vdo": pd.DataFrame(
            vdo_rows
        ),
        "alpha_grid": alpha_grid,
        "config": config,
    }


def solve_category_milp(
    candidate_table: pd.DataFrame,
    products: list[str],
    horizon: int,
    weekly_capacity: int,
    value_column: str,
) -> pd.DataFrame:
    candidates = candidate_table.reset_index(
        drop=True
    ).copy()
    variable_count = len(candidates)

    product_rows = [
        candidates["upc"].astype(str).eq(
            upc
        ).astype(float).to_numpy()
        for upc in products
    ]
    capacity_rows = [
        candidates[
            f"week_{week}"
        ].gt(0).astype(float).to_numpy()
        for week in range(
            1,
            horizon + 1,
        )
    ]
    matrix = np.vstack(
        [
            *product_rows,
            *capacity_rows,
        ]
    )
    lower = np.concatenate(
        [
            np.ones(
                len(products)
            ),
            np.full(
                horizon,
                -np.inf,
            ),
        ]
    )
    upper = np.concatenate(
        [
            np.ones(
                len(products)
            ),
            np.full(
                horizon,
                weekly_capacity,
            ),
        ]
    )
    constraints = LinearConstraint(
        matrix,
        lower,
        upper,
    )
    result = milp(
        c=-candidates[
            value_column
        ].to_numpy(
            dtype=float
        ),
        integrality=np.ones(
            variable_count,
            dtype=int,
        ),
        bounds=Bounds(
            np.zeros(
                variable_count
            ),
            np.ones(
                variable_count
            ),
        ),
        constraints=constraints,
        options={
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Category MILP failed: {result.message}"
        )

    selected = candidates.loc[
        result.x > 0.5
    ].copy()

    if selected["upc"].nunique() != len(
        products
    ):
        raise RuntimeError(
            "MILP did not select one schedule per product."
        )

    return selected


def solve_myopic_category(
    artifact: dict[str, Any],
    alpha: float,
    weekly_capacity: int,
) -> dict[str, np.ndarray]:
    products = artifact["products"]
    product_artifacts = artifact[
        "product_artifacts"
    ]
    config: PlanningConfig = artifact[
        "config"
    ]
    actions = np.asarray(
        config.actions,
        dtype=float,
    )
    inventories = {
        upc: 0.0
        for upc in products
    }
    cooldowns = {
        upc: 0
        for upc in products
    }
    promotions = {
        upc: 0
        for upc in products
    }
    schedules = {
        upc: []
        for upc in products
    }
    theta_means = {
        upc: np.average(
            product_artifacts[
                upc
            ]["theta"],
            axis=0,
            weights=product_artifacts[
                upc
            ]["weights"],
        )
        for upc in products
    }

    for _ in range(
        config.horizon
    ):
        proposals = []

        for upc in products:
            (
                elasticity,
                promotion_lift,
                displacement,
                persistence,
            ) = theta_means[upc]
            economics = product_artifacts[
                upc
            ]["economics"]
            no_promo_demand = (
                economics.base_demand
                * np.exp(
                    -displacement
                    * max(
                        inventories[upc],
                        0.0,
                    )
                )
            )
            no_promo_profit = (
                (
                    economics.regular_price
                    - economics.unit_cost
                )
                * no_promo_demand
            )

            if (
                cooldowns[upc] > 0
                or promotions[upc]
                >= config.max_promotions
            ):
                proposals.append(
                    (
                        -np.inf,
                        upc,
                        0.0,
                    )
                )
                continue

            action_results = []

            for depth in actions[
                actions > 1e-12
            ]:
                demand = (
                    economics.base_demand
                    * np.power(
                        1.0 - depth,
                        -elasticity,
                    )
                    * np.exp(
                        promotion_lift
                        - displacement
                        * max(
                            inventories[upc],
                            0.0,
                        )
                    )
                )
                price = (
                    economics.regular_price
                    * (
                        1.0 - depth
                    )
                )
                profit = (
                    (
                        price
                        - economics.unit_cost
                        + alpha * depth
                    )
                    * demand
                )
                action_results.append(
                    (
                        float(
                            profit
                            - no_promo_profit
                        ),
                        float(depth),
                    )
                )

            best_gain, best_depth = max(
                action_results,
                key=lambda item: (
                    item[0],
                    -item[1],
                ),
            )
            proposals.append(
                (
                    best_gain,
                    upc,
                    best_depth,
                )
            )

        selected = {
            upc
            for gain, upc, depth in sorted(
                proposals,
                reverse=True,
            )[:weekly_capacity]
            if gain > 0
            and depth > 1e-12
        }
        proposal_lookup = {
            upc: depth
            for gain, upc, depth in proposals
            if gain > 0
        }

        for upc in products:
            depth = (
                float(
                    proposal_lookup[upc]
                )
                if upc in selected
                else 0.0
            )
            schedules[upc].append(depth)
            persistence = theta_means[
                upc
            ][3]
            inventories[upc] = (
                persistence
                * max(
                    inventories[upc],
                    0.0,
                )
                + depth
            )

            if depth > 1e-12:
                promotions[upc] += 1
                cooldowns[upc] = (
                    config.cooldown
                )
            else:
                cooldowns[upc] = max(
                    cooldowns[upc] - 1,
                    0,
                )

    return {
        upc: np.asarray(
            schedule,
            dtype=float,
        )
        for upc, schedule in schedules.items()
    }


def evaluate_nominal_schedule(
    artifact: dict[str, Any],
    upc: str,
    schedule: np.ndarray,
    alpha: float,
) -> float:
    key = tuple(
        np.asarray(
            schedule,
            dtype=float,
        ).tolist()
    )
    schedule_lookup = artifact[
        "schedule_lookup"
    ]
    product = artifact[
        "product_artifacts"
    ][upc]

    if key in schedule_lookup:
        schedule_index = schedule_lookup[
            key
        ]

        return float(
            product["nominal_base"][
                schedule_index
            ]
            + alpha
            * product[
                "nominal_exposure"
            ][
                schedule_index
            ]
        )

    config: PlanningConfig = artifact[
        "config"
    ]
    values = []

    for draw in product["theta"]:
        (
            elasticity,
            promotion_lift,
            displacement,
            persistence,
        ) = draw
        economics = product[
            "economics"
        ]
        inventory = 0.0
        value = 0.0

        for time_index, depth in enumerate(
            schedule
        ):
            demand = (
                economics.base_demand
                * np.power(
                    max(
                        1.0 - depth,
                        1e-8,
                    ),
                    -elasticity,
                )
                * np.exp(
                    promotion_lift
                    * float(
                        depth > 1e-12
                    )
                    - displacement
                    * max(
                        inventory,
                        0.0,
                    )
                )
            )
            price = (
                economics.regular_price
                * (
                    1.0 - depth
                )
            )
            value += (
                config.discount_factor
                ** time_index
            ) * (
                price
                - economics.unit_cost
                + alpha * depth
            ) * demand
            inventory = (
                persistence
                * max(
                    inventory,
                    0.0,
                )
                + depth
            )

        for washout_index in range(
            config.washout_horizon
        ):
            time_index = (
                len(
                    schedule
                )
                + washout_index
            )
            demand = (
                economics.base_demand
                * np.exp(
                    -displacement
                    * max(
                        inventory,
                        0.0,
                    )
                )
            )
            value += (
                config.discount_factor
                ** time_index
            ) * (
                economics.regular_price
                - economics.unit_cost
            ) * demand
            inventory = (
                persistence
                * max(
                    inventory,
                    0.0,
                )
            )

        values.append(value)

    return float(
        np.average(
            values,
            weights=product["weights"],
        )
    )


def build_category_results(
    artifact: dict[str, Any],
    alpha_grid: np.ndarray,
    weekly_capacity: int,
) -> dict[str, pd.DataFrame]:
    products = artifact["products"]
    schedules = artifact["schedules"]
    frontier = artifact[
        "frontier_table"
    ].copy()
    schedule_lookup = artifact[
        "schedule_lookup"
    ]
    config: PlanningConfig = artifact[
        "config"
    ]
    calendar_rows = []
    summary_rows = []
    product_value_rows = []

    for raw_alpha in alpha_grid:
        alpha = float(raw_alpha)
        candidates = frontier.copy()
        candidates["nominal_value"] = (
            candidates["nominal_base"]
            + alpha
            * candidates[
                "nominal_exposure"
            ]
        )
        dynamic_selected = solve_category_milp(
            candidates,
            products,
            config.horizon,
            weekly_capacity,
            "nominal_value",
        )
        myopic_schedules = solve_myopic_category(
            artifact,
            alpha,
            weekly_capacity,
        )
        policies = {
            "dynamic": {
                str(row.upc): schedules[
                    int(row.schedule_id)
                ]
                for row in dynamic_selected.itertuples(
                    index=False
                )
            },
            "myopic": myopic_schedules,
        }

        for policy, policy_schedules in policies.items():
            total_value = 0.0
            total_promotions = 0
            total_depth = 0.0
            supported_schedule_count = 0
            promoted_products = 0
            weekly_promotions = np.zeros(
                config.horizon,
                dtype=int,
            )

            for upc in products:
                schedule = np.asarray(
                    policy_schedules[upc],
                    dtype=float,
                )
                value = evaluate_nominal_schedule(
                    artifact,
                    upc,
                    schedule,
                    alpha,
                )
                promotion_count = int(
                    np.sum(
                        schedule > 1e-12
                    )
                )
                total_value += value
                total_promotions += (
                    promotion_count
                )
                total_depth += float(
                    schedule.sum()
                )
                weekly_promotions += (
                    schedule > 1e-12
                ).astype(int)

                schedule_id = schedule_lookup.get(
                    tuple(
                        schedule.tolist()
                    ),
                    -1,
                )
                support_match = frontier.loc[
                    frontier["upc"].astype(
                        str
                    ).eq(upc)
                    & frontier[
                        "schedule_id"
                    ].eq(
                        schedule_id
                    ),
                    "well_supported",
                ]
                supported = bool(
                    support_match.iloc[0]
                ) if not support_match.empty else False

                if promotion_count > 0:
                    promoted_products += 1
                    supported_schedule_count += int(
                        supported
                    )

                product_value_rows.append(
                    {
                        "alpha": alpha,
                        "policy": policy,
                        "upc": upc,
                        "product_name": (
                            artifact[
                                "product_names"
                            ][upc]
                        ),
                        "value": value,
                        "promotion_count": (
                            promotion_count
                        ),
                        "total_depth": float(
                            schedule.sum()
                        ),
                        "schedule_supported": (
                            supported
                        ),
                    }
                )

                for week_index, depth in enumerate(
                    schedule,
                    start=1,
                ):
                    calendar_rows.append(
                        {
                            "alpha": alpha,
                            "policy": policy,
                            "upc": upc,
                            "product_name": (
                                artifact[
                                    "product_names"
                                ][upc]
                            ),
                            "week": (
                                week_index
                            ),
                            "discount_depth": float(
                                depth
                            ),
                        }
                    )

            summary_rows.append(
                {
                    "alpha": alpha,
                    "policy": policy,
                    "category_value": (
                        total_value
                    ),
                    "promotion_count": (
                        total_promotions
                    ),
                    "total_discount_depth": (
                        total_depth
                    ),
                    "maximum_weekly_promotions": int(
                        weekly_promotions.max()
                    ),
                    "slot_utilization": float(
                        weekly_promotions.sum()
                        / (
                            config.horizon
                            * weekly_capacity
                        )
                    ),
                    "supported_product_schedule_share": (
                        supported_schedule_count
                        / promoted_products
                        if promoted_products > 0
                        else 1.0
                    ),
                }
            )

    calendars = pd.DataFrame(
        calendar_rows
    )
    summary = pd.DataFrame(
        summary_rows
    )
    product_values = pd.DataFrame(
        product_value_rows
    )

    return {
        "category_calendars": calendars,
        "category_summary": summary,
        "category_product_values": (
            product_values
        ),
    }


def _reduce_distribution(
    theta: np.ndarray,
    weights: np.ndarray,
    scenario_count: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    mean = np.average(
        theta,
        axis=0,
        weights=weights,
    )
    variance = np.average(
        (
            theta - mean
        ) ** 2,
        axis=0,
        weights=weights,
    )
    scale = np.sqrt(
        variance
    )
    scale = np.where(
        scale > 1e-10,
        scale,
        1.0,
    )
    standardized = (
        theta - mean
    ) / scale
    scenario_count = min(
        scenario_count,
        len(theta),
    )
    kmeans = KMeans(
        n_clusters=scenario_count,
        random_state=random_seed,
        n_init=20,
    )
    kmeans.fit(
        standardized,
        sample_weight=weights,
    )
    indices = []
    reduced_weights = []

    for cluster_id in range(
        scenario_count
    ):
        members = np.flatnonzero(
            kmeans.labels_ == cluster_id
        )
        center = kmeans.cluster_centers_[
            cluster_id
        ]
        distance = np.linalg.norm(
            standardized[members]
            - center,
            axis=1,
        )
        indices.append(
            int(
                members[
                    np.argmin(distance)
                ]
            )
        )
        reduced_weights.append(
            float(
                weights[members].sum()
            )
        )

    indices_array = np.asarray(
        indices,
        dtype=int,
    )
    reduced_weight_array = np.asarray(
        reduced_weights,
        dtype=float,
    )
    reduced_weight_array = (
        reduced_weight_array
        / reduced_weight_array.sum()
    )
    reduced_standardized = standardized[
        indices_array
    ]

    return {
        "indices": indices_array,
        "weights": (
            reduced_weight_array
        ),
        "standardized": (
            reduced_standardized
        ),
        "cost": cdist(
            reduced_standardized,
            reduced_standardized,
        ),
    }


def _equal_weight_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    cost = cdist(
        first,
        second,
    )
    rows, columns = linear_sum_assignment(
        cost
    )

    return float(
        cost[
            rows,
            columns,
        ].mean()
    )


def _calibrate_rho(
    standardized: np.ndarray,
    weights: np.ndarray,
    random_seed: int,
    replications: int,
    quantile: float,
) -> float:
    rng = np.random.default_rng(
        random_seed
    )
    sample_size = min(
        40,
        max(
            10,
            len(standardized),
        ),
    )
    distances = []

    for _ in range(replications):
        first = rng.choice(
            len(standardized),
            size=sample_size,
            replace=True,
            p=weights,
        )
        second = rng.choice(
            len(standardized),
            size=sample_size,
            replace=True,
            p=weights,
        )
        distances.append(
            _equal_weight_wasserstein(
                standardized[first],
                standardized[second],
            )
        )

    return float(
        np.quantile(
            distances,
            quantile,
        )
    )


def _worst_case_expectation(
    values: np.ndarray,
    weights: np.ndarray,
    cost: np.ndarray,
    rho: float,
) -> float:
    nominal = float(
        np.dot(
            weights,
            values,
        )
    )

    if rho <= 1e-12:
        return nominal

    value_range = float(
        values.max() - values.min()
    )
    positive_cost = cost[
        cost > 1e-12
    ]

    if (
        value_range <= 1e-12
        or len(positive_cost) == 0
    ):
        return float(
            values.min()
        )

    lambda_upper = (
        value_range
        / float(
            positive_cost.min()
        )
        + 1.0
    )

    def dual_value(
        multiplier: float,
    ) -> float:
        transported = (
            values[None, :]
            + multiplier * cost
        )
        source_minima = transported.min(
            axis=1
        )

        return float(
            -multiplier * rho
            + np.dot(
                weights,
                source_minima,
            )
        )

    result = minimize_scalar(
        lambda multiplier: -dual_value(
            float(multiplier)
        ),
        bounds=(
            0.0,
            lambda_upper,
        ),
        method="bounded",
    )

    return float(
        max(
            dual_value(0.0),
            dual_value(
                float(result.x)
            ),
            dual_value(
                lambda_upper
            ),
        )
    )


def category_dro_ablation(
    artifact: dict[str, Any],
    alpha: float,
    weekly_capacity: int,
    scenario_count: int = 20,
    radius_replications: int = 50,
    radius_quantile: float = 0.90,
    random_seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    products = artifact["products"]
    frontier = artifact[
        "frontier_table"
    ].copy()
    product_artifacts = artifact[
        "product_artifacts"
    ]
    config: PlanningConfig = artifact[
        "config"
    ]
    candidate_frames = []
    diagnostic_rows = []

    for product_index, upc in enumerate(
        products
    ):
        product = product_artifacts[upc]
        reduced = _reduce_distribution(
            product["theta"],
            product["weights"],
            scenario_count,
            random_seed + product_index,
        )
        rho = _calibrate_rho(
            reduced["standardized"],
            reduced["weights"],
            random_seed
            + 1000
            + product_index,
            radius_replications,
            radius_quantile,
        )
        candidate_rows = frontier.loc[
            frontier["upc"].astype(
                str
            ).eq(upc)
        ].copy()
        robust_values = []

        for row in candidate_rows.itertuples(
            index=False
        ):
            schedule_id = int(
                row.schedule_id
            )
            scenario_values = (
                product["base_matrix"][
                    schedule_id,
                    reduced["indices"],
                ].astype(float)
                + alpha
                * product[
                    "exposure_matrix"
                ][
                    schedule_id,
                    reduced["indices"],
                ].astype(float)
            )
            robust_values.append(
                _worst_case_expectation(
                    scenario_values,
                    reduced["weights"],
                    reduced["cost"],
                    rho,
                )
            )

        candidate_rows[
            "robust_value"
        ] = robust_values
        candidate_frames.append(
            candidate_rows
        )
        diagnostic_rows.append(
            {
                "upc": upc,
                "product_name": (
                    artifact[
                        "product_names"
                    ][upc]
                ),
                "rho": rho,
                "scenario_count": int(
                    len(
                        reduced["indices"]
                    )
                ),
            }
        )

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )
    selected = solve_category_milp(
        candidates,
        products,
        config.horizon,
        weekly_capacity,
        "robust_value",
    )
    rows = []

    for row in selected.itertuples(
        index=False
    ):
        upc = str(row.upc)
        schedule_id = int(
            row.schedule_id
        )
        product = product_artifacts[upc]
        nominal_value = float(
            product["nominal_base"][
                schedule_id
            ]
            + alpha
            * product[
                "nominal_exposure"
            ][
                schedule_id
            ]
        )
        rows.append(
            {
                "upc": upc,
                "product_name": (
                    artifact[
                        "product_names"
                    ][upc]
                ),
                "schedule_id": (
                    schedule_id
                ),
                "robust_objective": float(
                    row.robust_value
                ),
                "nominal_evaluation": (
                    nominal_value
                ),
                "promotion_count": int(
                    artifact[
                        "promotion_count"
                    ][
                        schedule_id
                    ]
                ),
                "total_depth": float(
                    artifact[
                        "total_depth"
                    ][
                        schedule_id
                    ]
                ),
                "well_supported": bool(
                    row.well_supported
                ),
            }
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            diagnostic_rows
        ),
    )

# ================================================================
# V2: SUPPORT-CONSTRAINED ACTIONS AND PRODUCT SCHEDULES
# ================================================================


@dataclass(frozen=True)
class SupportedActionConfig:
    bin_width: float = 0.05
    minimum_depth: float = 0.03
    maximum_depth: float = 0.60
    minimum_observations: int = 15
    minimum_panels: int = 3
    maximum_positive_actions: int = 3
    center_rounding: int = 2
    matching_tolerance: float = 0.03


def build_product_supported_action_sets(
    history: pd.DataFrame,
    selected_products: list[str],
    product_names: dict[str, str],
    config: SupportedActionConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, tuple[float, ...]],
]:
    calibration = history.loc[
        history[
            "sample_period"
        ].eq(
            "calibration"
        )
    ].copy()
    calibration[
        "upc"
    ] = calibration[
        "upc"
    ].astype(str)

    promotion_rows = calibration.loc[
        calibration[
            "promotion_indicator"
        ].eq(1)
        & calibration[
            "discount_depth_model"
        ].between(
            config.minimum_depth,
            config.maximum_depth,
            inclusive="both",
        )
    ].copy()

    promotion_rows[
        "support_bin_center"
    ] = (
        np.round(
            promotion_rows[
                "discount_depth_model"
            ]
            / config.bin_width
        )
        * config.bin_width
    ).clip(
        config.minimum_depth,
        config.maximum_depth,
    )

    rows: list[
        dict[str, Any]
    ] = []
    action_sets: dict[
        str,
        tuple[float, ...],
    ] = {}

    for raw_upc in selected_products:
        upc = str(
            raw_upc
        )
        product_rows = calibration.loc[
            calibration[
                "upc"
            ].eq(
                upc
            )
        ]
        product_promotions = (
            promotion_rows.loc[
                promotion_rows[
                    "upc"
                ].eq(
                    upc
                )
            ]
        )
        regular_rows = product_rows.loc[
            product_rows[
                "promotion_indicator"
            ].eq(0)
        ]

        rows.append(
            {
                "upc": upc,
                "product_name": (
                    product_names.get(
                        upc,
                        upc,
                    )
                ),
                "action": 0.0,
                "bin_center": 0.0,
                "support_count": int(
                    len(
                        regular_rows
                    )
                ),
                "support_panels": int(
                    regular_rows[
                        "store_upc"
                    ].nunique()
                ),
                "support_stores": int(
                    regular_rows[
                        "store"
                    ].nunique()
                ),
                "depth_q10": 0.0,
                "depth_q50": 0.0,
                "depth_q90": 0.0,
                "supported": True,
                "selected_for_grid": True,
                "selection_reason": (
                    "reference_action"
                ),
            }
        )

        clusters = []

        for (
            bin_center,
            cluster,
        ) in product_promotions.groupby(
            "support_bin_center",
            observed=True,
        ):
            count = int(
                len(
                    cluster
                )
            )
            panels = int(
                cluster[
                    "store_upc"
                ].nunique()
            )
            stores = int(
                cluster[
                    "store"
                ].nunique()
            )
            action = float(
                np.round(
                    cluster[
                        "discount_depth_model"
                    ].median(),
                    config.center_rounding,
                )
            )
            supported = bool(
                count
                >= config.minimum_observations
                and panels
                >= config.minimum_panels
            )

            clusters.append(
                {
                    "upc": upc,
                    "product_name": (
                        product_names.get(
                            upc,
                            upc,
                        )
                    ),
                    "action": action,
                    "bin_center": float(
                        bin_center
                    ),
                    "support_count": (
                        count
                    ),
                    "support_panels": (
                        panels
                    ),
                    "support_stores": (
                        stores
                    ),
                    "depth_q10": float(
                        cluster[
                            "discount_depth_model"
                        ].quantile(
                            0.10
                        )
                    ),
                    "depth_q50": float(
                        cluster[
                            "discount_depth_model"
                        ].median()
                    ),
                    "depth_q90": float(
                        cluster[
                            "discount_depth_model"
                        ].quantile(
                            0.90
                        )
                    ),
                    "supported": (
                        supported
                    ),
                    "selected_for_grid": False,
                    "selection_reason": (
                        "below_support_threshold"
                    ),
                }
            )

        supported_indices = [
            index
            for index, row in enumerate(
                clusters
            )
            if row[
                "supported"
            ]
        ]
        selected_indices: list[
            int
        ] = []

        if supported_indices:
            first = max(
                supported_indices,
                key=lambda index: (
                    clusters[
                        index
                    ][
                        "support_count"
                    ],
                    clusters[
                        index
                    ][
                        "support_panels"
                    ],
                ),
            )
            selected_indices.append(
                first
            )

            while (
                len(
                    selected_indices
                )
                < config.maximum_positive_actions
                and len(
                    selected_indices
                )
                < len(
                    supported_indices
                )
            ):
                remaining = [
                    index
                    for index in (
                        supported_indices
                    )
                    if index
                    not in selected_indices
                ]
                next_index = max(
                    remaining,
                    key=lambda index: (
                        min(
                            abs(
                                clusters[
                                    index
                                ][
                                    "action"
                                ]
                                - clusters[
                                    chosen
                                ][
                                    "action"
                                ]
                            )
                            for chosen in (
                                selected_indices
                            )
                        ),
                        np.log1p(
                            clusters[
                                index
                            ][
                                "support_count"
                            ]
                        ),
                    ),
                )
                selected_indices.append(
                    next_index
                )

        for index, row in enumerate(
            clusters
        ):
            if index in selected_indices:
                row[
                    "selected_for_grid"
                ] = True
                row[
                    "selection_reason"
                ] = (
                    "supported_representative"
                )
            rows.append(
                row
            )

        selected_actions = sorted(
            {
                float(
                    clusters[
                        index
                    ][
                        "action"
                    ]
                )
                for index in (
                    selected_indices
                )
            }
        )
        action_sets[
            upc
        ] = tuple(
            [
                0.0,
                *selected_actions,
            ]
        )

    return (
        pd.DataFrame(
            rows
        ),
        action_sets,
    )


def enumerate_product_schedules(
    actions: tuple[float, ...],
    horizon: int,
    cooldown: int,
    max_promotions: int,
) -> np.ndarray:
    action_array = np.asarray(
        sorted(
            set(
                float(
                    action
                )
                for action in (
                    actions
                )
            )
        ),
        dtype=float,
    )

    if (
        len(
            action_array
        )
        == 0
        or not np.isclose(
            action_array[
                0
            ],
            0.0,
        )
    ):
        action_array = np.insert(
            action_array,
            0,
            0.0,
        )

    schedules: list[
        tuple[float, ...]
    ] = []

    def recurse(
        time_index: int,
        cooldown_remaining: int,
        promotion_count: int,
        path: list[float],
    ) -> None:
        if time_index == horizon:
            schedules.append(
                tuple(
                    path
                )
            )
            return

        if cooldown_remaining > 0:
            feasible_actions = np.array(
                [
                    0.0
                ],
                dtype=float,
            )
        else:
            feasible_actions = np.asarray(
                [
                    action
                    for action in (
                        action_array
                    )
                    if (
                        action <= 1e-12
                        or promotion_count
                        < max_promotions
                    )
                ],
                dtype=float,
            )

        for depth in feasible_actions:
            promoted = bool(
                depth
                > 1e-12
            )
            path.append(
                float(
                    depth
                )
            )
            recurse(
                time_index + 1,
                (
                    cooldown
                    if promoted
                    else max(
                        cooldown_remaining
                        - 1,
                        0,
                    )
                ),
                promotion_count
                + int(
                    promoted
                ),
                path,
            )
            path.pop()

    recurse(
        0,
        0,
        0,
        [],
    )

    return np.asarray(
        schedules,
        dtype=float,
    )


def upper_envelope_indices(
    indices: np.ndarray,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    alpha_min: float,
    alpha_max: float,
    tolerance: float = 1e-10,
) -> np.ndarray:
    ordered = sorted(
        [
            int(
                index
            )
            for index in (
                indices
            )
        ],
        key=lambda index: (
            float(
                slopes[
                    index
                ]
            ),
            -float(
                intercepts[
                    index
                ]
            ),
            index,
        ),
    )

    if len(
        ordered
    ) <= 1:
        return np.asarray(
            ordered,
            dtype=int,
        )

    unique_lines: list[
        int
    ] = []

    for index in ordered:
        if (
            unique_lines
            and abs(
                float(
                    slopes[
                        index
                    ]
                    - slopes[
                        unique_lines[
                            -1
                        ]
                    ]
                )
            )
            <= tolerance
        ):
            previous = unique_lines[
                -1
            ]
            if (
                intercepts[
                    index
                ]
                > intercepts[
                    previous
                ]
                + tolerance
            ):
                unique_lines[
                    -1
                ] = index
        else:
            unique_lines.append(
                index
            )

    hull_indices: list[
        int
    ] = []
    hull_starts: list[
        float
    ] = []

    for index in unique_lines:
        start = -np.inf

        while hull_indices:
            previous = hull_indices[
                -1
            ]
            denominator = float(
                slopes[
                    index
                ]
                - slopes[
                    previous
                ]
            )
            start = float(
                (
                    intercepts[
                        previous
                    ]
                    - intercepts[
                        index
                    ]
                )
                / denominator
            )

            if (
                len(
                    hull_indices
                )
                == 1
                or start
                > hull_starts[
                    -1
                ]
                + tolerance
            ):
                break

            hull_indices.pop()
            hull_starts.pop()

        if not hull_indices:
            start = -np.inf

        hull_indices.append(
            index
        )
        hull_starts.append(
            start
        )

    retained = []

    for position, index in enumerate(
        hull_indices
    ):
        interval_start = hull_starts[
            position
        ]
        interval_end = (
            hull_starts[
                position + 1
            ]
            if position + 1
            < len(
                hull_starts
            )
            else np.inf
        )

        if (
            interval_end
            >= alpha_min
            - tolerance
            and interval_start
            <= alpha_max
            + tolerance
        ):
            retained.append(
                index
            )

    return np.asarray(
        sorted(
            set(
                retained
            )
        ),
        dtype=int,
    )


def _support_for_schedule_v2(
    upc: str,
    schedule: np.ndarray,
    support_table: pd.DataFrame,
    matching_tolerance: float,
) -> tuple[
    int,
    bool,
]:
    positive_actions = np.unique(
        schedule[
            schedule
            > 1e-12
        ]
    )

    if len(
        positive_actions
    ) == 0:
        regular_count = support_table.loc[
            support_table[
                "upc"
            ].astype(
                str
            ).eq(
                upc
            )
            & np.isclose(
                support_table[
                    "action"
                ],
                0.0,
            ),
            "support_count",
        ]

        return (
            int(
                regular_count.max()
            )
            if not regular_count.empty
            else 0,
            True,
        )

    selected_rows = support_table.loc[
        support_table[
            "upc"
        ].astype(
            str
        ).eq(
            upc
        )
        & support_table[
            "selected_for_grid"
        ].astype(
            bool
        )
        & support_table[
            "action"
        ].gt(
            0
        )
    ].copy()

    if selected_rows.empty:
        return (
            0,
            False,
        )

    minimum_count = np.inf
    all_supported = True

    for action in positive_actions:
        distances = (
            selected_rows[
                "action"
            ]
            .sub(
                float(
                    action
                )
            )
            .abs()
        )
        closest_index = distances.idxmin()
        closest = selected_rows.loc[
            closest_index
        ]
        close_enough = bool(
            distances.loc[
                closest_index
            ]
            <= matching_tolerance
        )
        action_supported = bool(
            close_enough
            and closest[
                "supported"
            ]
        )
        all_supported &= (
            action_supported
        )
        minimum_count = min(
            minimum_count,
            int(
                closest[
                    "support_count"
                ]
            )
            if action_supported
            else 0,
        )

    return (
        int(
            minimum_count
        ),
        bool(
            all_supported
        ),
    )


def _myopic_schedule_v2(
    theta: np.ndarray,
    weights: np.ndarray,
    economics: ProductEconomics,
    actions: tuple[float, ...],
    alpha: float,
    horizon: int,
    cooldown: int,
    max_promotions: int,
) -> np.ndarray:
    (
        elasticity,
        promotion_lift,
        displacement,
        persistence,
    ) = np.average(
        theta,
        axis=0,
        weights=weights,
    )
    action_array = np.asarray(
        actions,
        dtype=float,
    )
    inventory = 0.0
    cooldown_remaining = 0
    promotion_count = 0
    schedule = []

    for _ in range(
        horizon
    ):
        if cooldown_remaining > 0:
            feasible_actions = np.asarray(
                [
                    0.0
                ],
                dtype=float,
            )
        else:
            feasible_actions = np.asarray(
                [
                    action
                    for action in (
                        action_array
                    )
                    if (
                        action <= 1e-12
                        or promotion_count
                        < max_promotions
                    )
                ],
                dtype=float,
            )

        action_values = []

        for depth in feasible_actions:
            demand = (
                economics.base_demand
                * np.power(
                    max(
                        1.0
                        - depth,
                        1e-8,
                    ),
                    -elasticity,
                )
                * np.exp(
                    promotion_lift
                    * float(
                        depth
                        > 1e-12
                    )
                    - displacement
                    * max(
                        inventory,
                        0.0,
                    )
                )
            )
            price = (
                economics.regular_price
                * (
                    1.0
                    - depth
                )
            )
            profit = (
                (
                    price
                    - economics.unit_cost
                    + alpha
                    * depth
                )
                * demand
            )
            action_values.append(
                (
                    float(
                        profit
                    ),
                    float(
                        depth
                    ),
                )
            )

        _, selected_depth = max(
            action_values,
            key=lambda item: (
                item[
                    0
                ],
                -item[
                    1
                ],
            ),
        )
        schedule.append(
            selected_depth
        )
        inventory = (
            persistence
            * max(
                inventory,
                0.0,
            )
            + selected_depth
        )

        if selected_depth > 1e-12:
            promotion_count += 1
            cooldown_remaining = (
                cooldown
            )
        else:
            cooldown_remaining = max(
                cooldown_remaining
                - 1,
                0,
            )

    return np.asarray(
        schedule,
        dtype=float,
    )


def build_product_schedule_artifact_v2(
    product_draws: pd.DataFrame,
    product_summary: pd.DataFrame,
    support_table: pd.DataFrame,
    product_action_sets: dict[
        str,
        tuple[float, ...],
    ],
    planning_config: PlanningConfig,
    support_config: SupportedActionConfig,
    grid_name: str,
) -> dict[str, Any]:
    draws_frame = product_draws.copy()
    draws_frame[
        "upc"
    ] = draws_frame[
        "upc"
    ].astype(
        str
    )
    summary = product_summary.copy()
    summary[
        "upc"
    ] = summary[
        "upc"
    ].astype(
        str
    )
    support = support_table.copy()
    support[
        "upc"
    ] = support[
        "upc"
    ].astype(
        str
    )

    for boolean_column in [
        "supported",
        "selected_for_grid",
    ]:
        if (
            boolean_column
            in support.columns
            and support[
                boolean_column
            ].dtype
            == object
        ):
            support[
                boolean_column
            ] = (
                support[
                    boolean_column
                ]
                .astype(
                    str
                )
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                    }
                )
                .fillna(
                    False
                )
            )

    products = summary[
        "upc"
    ].tolist()
    product_names = (
        summary.set_index(
            "upc"
        )[
            "product_name"
        ]
        .astype(
            str
        )
        .to_dict()
    )
    alpha_grid = np.round(
        np.arange(
            planning_config.alpha_min,
            planning_config.alpha_max
            + 0.5
            * planning_config.alpha_step,
            planning_config.alpha_step,
        ),
        10,
    )

    product_artifacts = {}
    frontier_rows = []
    escalation_rows = []
    policy_rows = []
    vdo_rows = []

    for upc in products:
        actions = tuple(
            sorted(
                set(
                    float(
                        action
                    )
                    for action in (
                        product_action_sets[
                            upc
                        ]
                    )
                )
            )
        )
        schedules = enumerate_product_schedules(
            actions=actions,
            horizon=(
                planning_config.horizon
            ),
            cooldown=(
                planning_config.cooldown
            ),
            max_promotions=(
                planning_config.max_promotions
            ),
        )
        promotion_count = (
            schedules
            > 1e-12
        ).sum(
            axis=1
        )
        total_depth = schedules.sum(
            axis=1
        )
        occupancy = (
            schedules
            > 1e-12
        ).astype(
            int
        )
        lookup = {
            tuple(
                schedule.tolist()
            ): int(
                schedule_index
            )
            for schedule_index, schedule in enumerate(
                schedules
            )
        }

        product_draw_subset = draws_frame.loc[
            draws_frame[
                "upc"
            ].eq(
                upc
            )
        ].copy()
        weights = product_draw_subset[
            "draw_weight"
        ].to_numpy(
            dtype=float
        )
        weights = weights / weights.sum()
        theta = product_draw_subset[
            PARAMETER_COLUMNS
        ].to_numpy(
            dtype=float
        )
        economics = ProductEconomics(
            base_demand=float(
                product_draw_subset[
                    "base_demand"
                ].median()
            ),
            regular_price=float(
                product_draw_subset[
                    "regular_price"
                ].median()
            ),
            unit_cost=float(
                product_draw_subset[
                    "unit_cost"
                ].median()
            ),
        )
        (
            base_matrix,
            exposure_matrix,
        ) = compute_value_matrices(
            schedules=schedules,
            theta=theta,
            economics=economics,
            discount_factor=(
                planning_config.discount_factor
            ),
            washout_horizon=(
                planning_config.washout_horizon
            ),
        )
        nominal_base = (
            base_matrix.astype(
                float
            )
            @ weights
        )
        nominal_exposure = (
            exposure_matrix.astype(
                float
            )
            @ weights
        )

        occupancy_groups = {}

        for schedule_index, pattern in enumerate(
            occupancy
        ):
            occupancy_groups.setdefault(
                tuple(
                    pattern.tolist()
                ),
                [],
            ).append(
                schedule_index
            )

        retained_indices = []

        for indices in occupancy_groups.values():
            retained_indices.extend(
                upper_envelope_indices(
                    indices=np.asarray(
                        indices,
                        dtype=int,
                    ),
                    intercepts=(
                        nominal_base
                    ),
                    slopes=(
                        nominal_exposure
                    ),
                    alpha_min=(
                        planning_config.alpha_min
                    ),
                    alpha_max=(
                        planning_config.alpha_max
                    ),
                ).tolist()
            )

        retained_indices = np.asarray(
            sorted(
                set(
                    retained_indices
                )
            ),
            dtype=int,
        )

        for schedule_index in retained_indices:
            (
                minimum_support,
                schedule_supported,
            ) = _support_for_schedule_v2(
                upc=upc,
                schedule=schedules[
                    schedule_index
                ],
                support_table=(
                    support
                ),
                matching_tolerance=(
                    support_config.matching_tolerance
                ),
            )
            row = {
                "grid_name": (
                    grid_name
                ),
                "upc": upc,
                "product_name": (
                    product_names[
                        upc
                    ]
                ),
                "schedule_id": int(
                    schedule_index
                ),
                "candidate_id": (
                    f"{grid_name}|"
                    f"{upc}|"
                    f"{schedule_index}"
                ),
                "nominal_base": float(
                    nominal_base[
                        schedule_index
                    ]
                ),
                "nominal_exposure": float(
                    nominal_exposure[
                        schedule_index
                    ]
                ),
                "promotion_count": int(
                    promotion_count[
                        schedule_index
                    ]
                ),
                "total_depth": float(
                    total_depth[
                        schedule_index
                    ]
                ),
                "minimum_action_support": int(
                    minimum_support
                ),
                "schedule_supported": bool(
                    schedule_supported
                ),
                "occupancy_pattern": "".join(
                    str(
                        int(
                            value
                        )
                    )
                    for value in (
                        occupancy[
                            schedule_index
                        ]
                    )
                ),
            }

            for week, depth in enumerate(
                schedules[
                    schedule_index
                ],
                start=1,
            ):
                row[
                    f"week_{week}"
                ] = float(
                    depth
                )

            frontier_rows.append(
                row
            )

        dynamic_indices = []
        myopic_indices = []

        for alpha in alpha_grid:
            values = (
                nominal_base
                + float(
                    alpha
                )
                * nominal_exposure
            )
            dynamic_index = _best_index(
                values=values,
                promotion_count=(
                    promotion_count
                ),
                total_depth=(
                    total_depth
                ),
            )
            myopic_schedule = _myopic_schedule_v2(
                theta=theta,
                weights=weights,
                economics=economics,
                actions=actions,
                alpha=float(
                    alpha
                ),
                horizon=(
                    planning_config.horizon
                ),
                cooldown=(
                    planning_config.cooldown
                ),
                max_promotions=(
                    planning_config.max_promotions
                ),
            )
            myopic_index = lookup[
                tuple(
                    myopic_schedule.tolist()
                )
            ]
            dynamic_indices.append(
                dynamic_index
            )
            myopic_indices.append(
                myopic_index
            )
            policy_rows.append(
                {
                    "grid_name": (
                        grid_name
                    ),
                    "upc": upc,
                    "product_name": (
                        product_names[
                            upc
                        ]
                    ),
                    "alpha": float(
                        alpha
                    ),
                    "dynamic_schedule_id": int(
                        dynamic_index
                    ),
                    "myopic_schedule_id": int(
                        myopic_index
                    ),
                    "dynamic_promotion_count": int(
                        promotion_count[
                            dynamic_index
                        ]
                    ),
                    "myopic_promotion_count": int(
                        promotion_count[
                            myopic_index
                        ]
                    ),
                    "dynamic_value": float(
                        values[
                            dynamic_index
                        ]
                    ),
                    "myopic_value": float(
                        values[
                            myopic_index
                        ]
                    ),
                    "value_of_dynamic_optimization": float(
                        values[
                            dynamic_index
                        ]
                        - values[
                            myopic_index
                        ]
                    ),
                }
            )

        dynamic_counts = promotion_count[
            np.asarray(
                dynamic_indices,
                dtype=int,
            )
        ]
        escalation = {
            "grid_name": (
                grid_name
            ),
            "upc": upc,
            "product_name": (
                product_names[
                    upc
                ]
            ),
        }

        for target in (
            1,
            2,
            4,
        ):
            positions = np.flatnonzero(
                dynamic_counts
                >= target
            )
            escalation[
                f"alpha_{target}_promotions"
            ] = (
                float(
                    alpha_grid[
                        positions[
                            0
                        ]
                    ]
                )
                if len(
                    positions
                )
                > 0
                else np.nan
            )

        escalation_rows.append(
            escalation
        )

        main_alpha = 2.24
        position = int(
            np.argmin(
                np.abs(
                    alpha_grid
                    - main_alpha
                )
            )
        )
        dynamic_index = dynamic_indices[
            position
        ]
        myopic_index = myopic_indices[
            position
        ]
        values = (
            nominal_base
            + main_alpha
            * nominal_exposure
        )
        vdo_rows.append(
            {
                "grid_name": (
                    grid_name
                ),
                "upc": upc,
                "product_name": (
                    product_names[
                        upc
                    ]
                ),
                "alpha": (
                    main_alpha
                ),
                "dynamic_schedule_id": int(
                    dynamic_index
                ),
                "myopic_schedule_id": int(
                    myopic_index
                ),
                "dynamic_value": float(
                    values[
                        dynamic_index
                    ]
                ),
                "myopic_value": float(
                    values[
                        myopic_index
                    ]
                ),
                "value_of_dynamic_optimization": float(
                    values[
                        dynamic_index
                    ]
                    - values[
                        myopic_index
                    ]
                ),
                "dynamic_promotion_count": int(
                    promotion_count[
                        dynamic_index
                    ]
                ),
                "myopic_promotion_count": int(
                    promotion_count[
                        myopic_index
                    ]
                ),
            }
        )

        product_artifacts[
            upc
        ] = {
            "actions": actions,
            "schedules": schedules,
            "promotion_count": (
                promotion_count
            ),
            "total_depth": (
                total_depth
            ),
            "occupancy": (
                occupancy
            ),
            "schedule_lookup": (
                lookup
            ),
            "no_promotion_index": int(
                np.flatnonzero(
                    promotion_count
                    == 0
                )[
                    0
                ]
            ),
            "theta": theta,
            "weights": weights,
            "economics": (
                economics
            ),
            "base_matrix": (
                base_matrix
            ),
            "exposure_matrix": (
                exposure_matrix
            ),
            "nominal_base": (
                nominal_base
            ),
            "nominal_exposure": (
                nominal_exposure
            ),
            "retained_indices": (
                retained_indices
            ),
        }

    return {
        "grid_name": (
            grid_name
        ),
        "products": (
            products
        ),
        "product_names": (
            product_names
        ),
        "product_action_sets": (
            product_action_sets
        ),
        "product_artifacts": (
            product_artifacts
        ),
        "frontier_table": pd.DataFrame(
            frontier_rows
        ),
        "escalation_table": pd.DataFrame(
            escalation_rows
        ),
        "policy_grid": pd.DataFrame(
            policy_rows
        ),
        "product_vdo": pd.DataFrame(
            vdo_rows
        ),
        "alpha_grid": (
            alpha_grid
        ),
        "planning_config": (
            planning_config
        ),
        "support_config": (
            support_config
        ),
    }

# ================================================================
# V2: CATEGORY CAPACITY AND CONTRACT-GENEROSITY SENSITIVITY
# ================================================================


def solve_category_milp_v2(
    candidate_table: pd.DataFrame,
    products: list[str],
    horizon: int,
    weekly_capacity: int,
    value_column: str,
) -> pd.DataFrame:
    candidates = candidate_table.reset_index(
        drop=True
    ).copy()

    if candidates.empty:
        raise ValueError(
            "The category candidate table is empty."
        )

    product_rows = [
        candidates[
            "upc"
        ].astype(
            str
        ).eq(
            upc
        ).astype(
            float
        ).to_numpy()
        for upc in (
            products
        )
    ]
    capacity_rows = [
        candidates[
            f"week_{week}"
        ].gt(
            0
        ).astype(
            float
        ).to_numpy()
        for week in range(
            1,
            horizon + 1,
        )
    ]
    matrix = np.vstack(
        [
            *product_rows,
            *capacity_rows,
        ]
    )
    lower = np.concatenate(
        [
            np.ones(
                len(
                    products
                )
            ),
            np.full(
                horizon,
                -np.inf,
            ),
        ]
    )
    upper = np.concatenate(
        [
            np.ones(
                len(
                    products
                )
            ),
            np.full(
                horizon,
                weekly_capacity,
            ),
        ]
    )

    result = milp(
        c=-candidates[
            value_column
        ].to_numpy(
            dtype=float
        ),
        integrality=np.ones(
            len(
                candidates
            ),
            dtype=int,
        ),
        bounds=Bounds(
            np.zeros(
                len(
                    candidates
                )
            ),
            np.ones(
                len(
                    candidates
                )
            ),
        ),
        constraints=LinearConstraint(
            matrix,
            lower,
            upper,
        ),
        options={
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Category MILP failed: "
            f"{result.message}"
        )

    selected = candidates.loc[
        result.x
        > 0.5
    ].copy()

    if selected[
        "upc"
    ].nunique() != len(
        products
    ):
        raise RuntimeError(
            "MILP did not select one schedule per product."
        )

    return selected


def solve_myopic_category_v2(
    artifact: dict[str, Any],
    alpha: float,
    weekly_capacity: int,
) -> dict[
    str,
    np.ndarray,
]:
    products = artifact[
        "products"
    ]
    product_artifacts = artifact[
        "product_artifacts"
    ]
    planning_config: PlanningConfig = artifact[
        "planning_config"
    ]

    inventories = {
        upc: 0.0
        for upc in products
    }
    cooldowns = {
        upc: 0
        for upc in products
    }
    promotion_counts = {
        upc: 0
        for upc in products
    }
    schedules = {
        upc: []
        for upc in products
    }
    theta_means = {
        upc: np.average(
            product_artifacts[
                upc
            ][
                "theta"
            ],
            axis=0,
            weights=(
                product_artifacts[
                    upc
                ][
                    "weights"
                ]
            ),
        )
        for upc in products
    }

    for _ in range(
        planning_config.horizon
    ):
        proposals = []

        for upc in products:
            product = product_artifacts[
                upc
            ]
            economics = product[
                "economics"
            ]
            (
                elasticity,
                promotion_lift,
                displacement,
                persistence,
            ) = theta_means[
                upc
            ]
            actions = np.asarray(
                product[
                    "actions"
                ],
                dtype=float,
            )

            no_promotion_demand = (
                economics.base_demand
                * np.exp(
                    -displacement
                    * max(
                        inventories[
                            upc
                        ],
                        0.0,
                    )
                )
            )
            no_promotion_profit = (
                (
                    economics.regular_price
                    - economics.unit_cost
                )
                * no_promotion_demand
            )

            if (
                cooldowns[
                    upc
                ]
                > 0
                or promotion_counts[
                    upc
                ]
                >= planning_config.max_promotions
                or not np.any(
                    actions
                    > 1e-12
                )
            ):
                proposals.append(
                    (
                        -np.inf,
                        upc,
                        0.0,
                    )
                )
                continue

            action_results = []

            for depth in actions[
                actions
                > 1e-12
            ]:
                demand = (
                    economics.base_demand
                    * np.power(
                        max(
                            1.0
                            - depth,
                            1e-8,
                        ),
                        -elasticity,
                    )
                    * np.exp(
                        promotion_lift
                        - displacement
                        * max(
                            inventories[
                                upc
                            ],
                            0.0,
                        )
                    )
                )
                price = (
                    economics.regular_price
                    * (
                        1.0
                        - depth
                    )
                )
                profit = (
                    (
                        price
                        - economics.unit_cost
                        + alpha
                        * depth
                    )
                    * demand
                )
                action_results.append(
                    (
                        float(
                            profit
                            - no_promotion_profit
                        ),
                        float(
                            depth
                        ),
                    )
                )

            best_gain, best_depth = max(
                action_results,
                key=lambda item: (
                    item[
                        0
                    ],
                    -item[
                        1
                    ],
                ),
            )
            proposals.append(
                (
                    best_gain,
                    upc,
                    best_depth,
                )
            )

        selected_products = {
            upc
            for (
                gain,
                upc,
                depth,
            ) in sorted(
                proposals,
                reverse=True,
            )[
                :weekly_capacity
            ]
            if (
                gain
                > 0
                and depth
                > 1e-12
            )
        }
        proposal_lookup = {
            upc: depth
            for (
                gain,
                upc,
                depth,
            ) in proposals
            if gain
            > 0
        }

        for upc in products:
            depth = (
                float(
                    proposal_lookup[
                        upc
                    ]
                )
                if upc
                in selected_products
                else 0.0
            )
            schedules[
                upc
            ].append(
                depth
            )
            persistence = theta_means[
                upc
            ][
                3
            ]
            inventories[
                upc
            ] = (
                persistence
                * max(
                    inventories[
                        upc
                    ],
                    0.0,
                )
                + depth
            )

            if depth > 1e-12:
                promotion_counts[
                    upc
                ] += 1
                cooldowns[
                    upc
                ] = (
                    planning_config.cooldown
                )
            else:
                cooldowns[
                    upc
                ] = max(
                    cooldowns[
                        upc
                    ]
                    - 1,
                    0,
                )

    return {
        upc: np.asarray(
            schedule,
            dtype=float,
        )
        for upc, schedule in (
            schedules.items()
        )
    }


def evaluate_nominal_schedule_v2(
    artifact: dict[str, Any],
    upc: str,
    schedule: np.ndarray,
    alpha: float,
) -> float:
    product = artifact[
        "product_artifacts"
    ][
        upc
    ]
    key = tuple(
        np.asarray(
            schedule,
            dtype=float,
        ).tolist()
    )

    if key in product[
        "schedule_lookup"
    ]:
        schedule_index = product[
            "schedule_lookup"
        ][
            key
        ]

        return float(
            product[
                "nominal_base"
            ][
                schedule_index
            ]
            + alpha
            * product[
                "nominal_exposure"
            ][
                schedule_index
            ]
        )

    planning_config: PlanningConfig = artifact[
        "planning_config"
    ]
    draw_values = []

    for draw in product[
        "theta"
    ]:
        (
            elasticity,
            promotion_lift,
            displacement,
            persistence,
        ) = draw
        economics = product[
            "economics"
        ]
        inventory = 0.0
        value = 0.0

        for time_index, depth in enumerate(
            schedule
        ):
            demand = (
                economics.base_demand
                * np.power(
                    max(
                        1.0
                        - depth,
                        1e-8,
                    ),
                    -elasticity,
                )
                * np.exp(
                    promotion_lift
                    * float(
                        depth
                        > 1e-12
                    )
                    - displacement
                    * max(
                        inventory,
                        0.0,
                    )
                )
            )
            price = (
                economics.regular_price
                * (
                    1.0
                    - depth
                )
            )
            value += (
                planning_config.discount_factor
                ** time_index
            ) * (
                price
                - economics.unit_cost
                + alpha
                * depth
            ) * demand
            inventory = (
                persistence
                * max(
                    inventory,
                    0.0,
                )
                + depth
            )

        for washout_index in range(
            planning_config.washout_horizon
        ):
            time_index = (
                len(
                    schedule
                )
                + washout_index
            )
            demand = (
                economics.base_demand
                * np.exp(
                    -displacement
                    * max(
                        inventory,
                        0.0,
                    )
                )
            )
            value += (
                planning_config.discount_factor
                ** time_index
            ) * (
                economics.regular_price
                - economics.unit_cost
            ) * demand
            inventory = (
                persistence
                * max(
                    inventory,
                    0.0,
                )
            )

        draw_values.append(
            value
        )

    return float(
        np.average(
            draw_values,
            weights=(
                product[
                    "weights"
                ]
            ),
        )
    )


def build_category_sensitivity_v2(
    artifacts: dict[
        str,
        dict[str, Any],
    ],
    alpha_grid: np.ndarray,
    capacities: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    calendar_rows = []
    summary_rows = []
    product_rows = []
    comparison_rows = []

    for grid_name, artifact in artifacts.items():
        products = artifact[
            "products"
        ]
        product_names = artifact[
            "product_names"
        ]
        planning_config: PlanningConfig = artifact[
            "planning_config"
        ]
        frontier = artifact[
            "frontier_table"
        ].copy()

        for raw_capacity in capacities:
            capacity = min(
                int(
                    raw_capacity
                ),
                len(
                    products
                ),
            )

            for raw_alpha in alpha_grid:
                alpha = float(
                    raw_alpha
                )
                candidates = frontier.copy()
                candidates[
                    "nominal_value"
                ] = (
                    candidates[
                        "nominal_base"
                    ]
                    + alpha
                    * candidates[
                        "nominal_exposure"
                    ]
                )
                dynamic_selected = (
                    solve_category_milp_v2(
                        candidate_table=(
                            candidates
                        ),
                        products=products,
                        horizon=(
                            planning_config.horizon
                        ),
                        weekly_capacity=(
                            capacity
                        ),
                        value_column=(
                            "nominal_value"
                        ),
                    )
                )
                myopic_schedules = (
                    solve_myopic_category_v2(
                        artifact=(
                            artifact
                        ),
                        alpha=alpha,
                        weekly_capacity=(
                            capacity
                        ),
                    )
                )
                policy_schedules = {
                    "dynamic": {
                        str(
                            row.upc
                        ): artifact[
                            "product_artifacts"
                        ][
                            str(
                                row.upc
                            )
                        ][
                            "schedules"
                        ][
                            int(
                                row.schedule_id
                            )
                        ]
                        for row in (
                            dynamic_selected.itertuples(
                                index=False
                            )
                        )
                    },
                    "myopic": (
                        myopic_schedules
                    ),
                }

                policy_product_counts = {}
                policy_active_sets = {}

                for (
                    policy,
                    schedules,
                ) in policy_schedules.items():
                    total_value = 0.0
                    total_promotions = 0
                    total_depth = 0.0
                    supported_promoted_products = 0
                    promoted_products = 0
                    weekly_occupancy = np.zeros(
                        planning_config.horizon,
                        dtype=int,
                    )
                    product_counts = {}
                    active_products = []

                    for upc in products:
                        schedule = np.asarray(
                            schedules[
                                upc
                            ],
                            dtype=float,
                        )
                        value = (
                            evaluate_nominal_schedule_v2(
                                artifact=(
                                    artifact
                                ),
                                upc=upc,
                                schedule=(
                                    schedule
                                ),
                                alpha=alpha,
                            )
                        )
                        promotion_count = int(
                            np.sum(
                                schedule
                                > 1e-12
                            )
                        )
                        product_counts[
                            upc
                        ] = (
                            promotion_count
                        )

                        if promotion_count > 0:
                            active_products.append(
                                upc
                            )

                        product = artifact[
                            "product_artifacts"
                        ][
                            upc
                        ]
                        schedule_id = product[
                            "schedule_lookup"
                        ].get(
                            tuple(
                                schedule.tolist()
                            ),
                            -1,
                        )
                        support_match = frontier.loc[
                            frontier[
                                "upc"
                            ].astype(
                                str
                            ).eq(
                                upc
                            )
                            & frontier[
                                "schedule_id"
                            ].eq(
                                schedule_id
                            ),
                            "schedule_supported",
                        ]
                        schedule_supported = bool(
                            support_match.iloc[
                                0
                            ]
                        ) if not support_match.empty else False

                        if promotion_count > 0:
                            promoted_products += 1
                            supported_promoted_products += int(
                                schedule_supported
                            )

                        total_value += (
                            value
                        )
                        total_promotions += (
                            promotion_count
                        )
                        total_depth += float(
                            schedule.sum()
                        )
                        weekly_occupancy += (
                            schedule
                            > 1e-12
                        ).astype(
                            int
                        )

                        product_rows.append(
                            {
                                "grid_name": (
                                    grid_name
                                ),
                                "weekly_capacity": (
                                    capacity
                                ),
                                "alpha": (
                                    alpha
                                ),
                                "policy": (
                                    policy
                                ),
                                "upc": (
                                    upc
                                ),
                                "product_name": (
                                    product_names[
                                        upc
                                    ]
                                ),
                                "value": (
                                    value
                                ),
                                "promotion_count": (
                                    promotion_count
                                ),
                                "total_depth": float(
                                    schedule.sum()
                                ),
                                "schedule_supported": (
                                    schedule_supported
                                ),
                            }
                        )

                        for week, depth in enumerate(
                            schedule,
                            start=1,
                        ):
                            calendar_rows.append(
                                {
                                    "grid_name": (
                                        grid_name
                                    ),
                                    "weekly_capacity": (
                                        capacity
                                    ),
                                    "alpha": (
                                        alpha
                                    ),
                                    "policy": (
                                        policy
                                    ),
                                    "upc": (
                                        upc
                                    ),
                                    "product_name": (
                                        product_names[
                                            upc
                                        ]
                                    ),
                                    "week": (
                                        week
                                    ),
                                    "discount_depth": float(
                                        depth
                                    ),
                                }
                            )

                    active_set = "|".join(
                        sorted(
                            active_products
                        )
                    )
                    summary_rows.append(
                        {
                            "grid_name": (
                                grid_name
                            ),
                            "weekly_capacity": (
                                capacity
                            ),
                            "alpha": (
                                alpha
                            ),
                            "policy": (
                                policy
                            ),
                            "category_value": float(
                                total_value
                            ),
                            "promotion_count": int(
                                total_promotions
                            ),
                            "total_discount_depth": float(
                                total_depth
                            ),
                            "maximum_weekly_promotions": int(
                                weekly_occupancy.max()
                            ),
                            "binding_weeks": int(
                                np.sum(
                                    weekly_occupancy
                                    >= capacity
                                )
                            ),
                            "slot_utilization": float(
                                weekly_occupancy.sum()
                                / (
                                    planning_config.horizon
                                    * capacity
                                )
                            ),
                            "active_product_count": int(
                                len(
                                    active_products
                                )
                            ),
                            "active_product_set": (
                                active_set
                            ),
                            "supported_promoted_product_share": (
                                supported_promoted_products
                                / promoted_products
                                if promoted_products
                                > 0
                                else 1.0
                            ),
                        }
                    )
                    policy_product_counts[
                        policy
                    ] = (
                        product_counts
                    )
                    policy_active_sets[
                        policy
                    ] = set(
                        active_products
                    )

                summary_frame = pd.DataFrame(
                    [
                        row
                        for row in (
                            summary_rows
                        )
                        if (
                            row[
                                "grid_name"
                            ]
                            == grid_name
                            and row[
                                "weekly_capacity"
                            ]
                            == capacity
                            and np.isclose(
                                row[
                                    "alpha"
                                ],
                                alpha,
                            )
                        )
                    ]
                ).set_index(
                    "policy"
                )
                dynamic_value = float(
                    summary_frame.loc[
                        "dynamic",
                        "category_value",
                    ]
                )
                myopic_value = float(
                    summary_frame.loc[
                        "myopic",
                        "category_value",
                    ]
                )
                dynamic_active = policy_active_sets[
                    "dynamic"
                ]
                myopic_active = policy_active_sets[
                    "myopic"
                ]
                count_disagreement = int(
                    sum(
                        abs(
                            policy_product_counts[
                                "dynamic"
                            ][
                                upc
                            ]
                            - policy_product_counts[
                                "myopic"
                            ][
                                upc
                            ]
                        )
                        for upc in products
                    )
                )
                dynamic_calendar = np.vstack(
                    [
                        policy_schedules[
                            "dynamic"
                        ][
                            upc
                        ]
                        for upc in products
                    ]
                )
                myopic_calendar = np.vstack(
                    [
                        policy_schedules[
                            "myopic"
                        ][
                            upc
                        ]
                        for upc in products
                    ]
                )
                timing_disagreement = int(
                    np.sum(
                        (
                            dynamic_calendar
                            > 1e-12
                        )
                        != (
                            myopic_calendar
                            > 1e-12
                        )
                    )
                )

                comparison_rows.append(
                    {
                        "grid_name": (
                            grid_name
                        ),
                        "weekly_capacity": (
                            capacity
                        ),
                        "alpha": (
                            alpha
                        ),
                        "value_of_dynamic_optimization": (
                            dynamic_value
                            - myopic_value
                        ),
                        "relative_VDO": (
                            (
                                dynamic_value
                                - myopic_value
                            )
                            / abs(
                                myopic_value
                            )
                            if abs(
                                myopic_value
                            )
                            > 1e-12
                            else np.nan
                        ),
                        "dynamic_active_product_set": "|".join(
                            sorted(
                                dynamic_active
                            )
                        ),
                        "myopic_active_product_set": "|".join(
                            sorted(
                                myopic_active
                            )
                        ),
                        "active_set_changed": bool(
                            dynamic_active
                            != myopic_active
                        ),
                        "active_set_symmetric_difference": int(
                            len(
                                dynamic_active
                                .symmetric_difference(
                                    myopic_active
                                )
                            )
                        ),
                        "product_promotion_count_disagreement": (
                            count_disagreement
                        ),
                        "product_week_timing_disagreement": (
                            timing_disagreement
                        ),
                        "dynamic_binding_weeks": int(
                            summary_frame.loc[
                                "dynamic",
                                "binding_weeks",
                            ]
                        ),
                        "myopic_binding_weeks": int(
                            summary_frame.loc[
                                "myopic",
                                "binding_weeks",
                            ]
                        ),
                        "dynamic_supported_share": float(
                            summary_frame.loc[
                                "dynamic",
                                "supported_promoted_product_share",
                            ]
                        ),
                        "myopic_supported_share": float(
                            summary_frame.loc[
                                "myopic",
                                "supported_promoted_product_share",
                            ]
                        ),
                    }
                )

    return {
        "calendars": pd.DataFrame(
            calendar_rows
        ),
        "summary": pd.DataFrame(
            summary_rows
        ),
        "product_values": pd.DataFrame(
            product_rows
        ),
        "comparisons": pd.DataFrame(
            comparison_rows
        ),
    }
