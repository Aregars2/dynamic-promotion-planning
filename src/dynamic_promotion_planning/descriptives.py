"""Descriptive-sample, support, and event-study helpers."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.dataset as ds
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal environments
    pa = None
    ds = None


SELECTED_SAMPLE_COLUMNS = [
    "store",
    "upc",
    "store_upc",
    "week",
    "week_start",
    "week_end",
    "move",
    "model_unit_price",
    "regular_price",
    "discount_depth",
    "price_imputed",
    "pricing_state",
    "promo_state",
    "post_promo",
    "gross_margin_pct_observed",
]

PREDICTION_CONTEXT_COLUMNS = [
    "store_upc",
    "upc",
    "week",
    "calendar_month",
    "scaled_time_trend",
    "thanksgiving_week",
    "christmas_week",
    "new_year_week",
    "easter_week",
    "log1p_lag_move",
    "log1p_lag_move_mean_4",
    "price_imputed_indicator",
    "mu_hat_regular_counterfactual",
]

EVENT_WINDOW_COLUMNS = [
    "event_id",
    "store_upc",
    "store",
    "upc",
    "week",
    "relative_week",
    "move",
    "model_unit_price",
    "regular_price",
    "price_imputed",
    "raw_baseline_sales",
    "raw_sales_index",
    "discount_depth",
    "promotion_indicator",
    "post_promotion_indicator",
    "split",
]


def normalize_identifier(series: pd.Series) -> pd.Series:
    """Normalize identifiers read from CSV or Parquet to comparable strings."""
    normalized = series.astype("string").str.strip()
    return normalized.str.replace(r"\.0$", "", regex=True)


def parse_boolean(series: pd.Series) -> pd.Series:
    """Parse Boolean, numeric, or string-valued indicators safely."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )
    if parsed.isna().any():
        bad_values = sorted(normalized.loc[parsed.isna()].dropna().unique())
        raise ValueError(f"Could not parse boolean values: {bad_values}")
    return parsed.astype(bool)


def coerce_values_for_arrow(
    values: list[str],
    arrow_type: pa.DataType,
) -> list:
    """Convert string identifiers to the physical type used in Parquet."""
    if pa is None:
        raise ImportError("pyarrow is required for filtered Parquet reads.")
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return [str(value) for value in values]
    if pa.types.is_integer(arrow_type):
        return [int(float(value)) for value in values]
    if pa.types.is_floating(arrow_type):
        return [float(value) for value in values]
    raise TypeError(f"Unsupported identifier type for filtered read: {arrow_type}")


def selection_fingerprint(
    source_data_path: Path,
    selected_upcs: list[str],
    selected_stores: list[str],
    eligible_panels: set[str],
) -> str:
    """Hash the selected-sample inputs used by the compact cache."""
    payload = {
        "source_path": str(source_data_path.resolve()),
        "source_mtime_ns": source_data_path.stat().st_mtime_ns,
        "selected_upcs": sorted(selected_upcs),
        "selected_stores": sorted(selected_stores),
        "eligible_panels": sorted(eligible_panels),
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return sha256(serialized).hexdigest()


def format_depths(values: pd.Series) -> str:
    """Format unique promotion depths for LaTeX-ready descriptive tables."""
    unique_values = sorted(pd.unique(values.dropna()))
    if not unique_values:
        return "None"
    return ", ".join(f"{100 * value:.0f}\\%" for value in unique_values)


def _read_selected_sample(
    source_data_path: Path,
    selected_upcs: list[str],
    selected_stores: list[str],
    eligible_panels: set[str],
) -> pd.DataFrame:
    if ds is None:
        raise ImportError("pyarrow is required to build the selected sample.")
    source_dataset = ds.dataset(source_data_path, format="parquet")
    missing = set(SELECTED_SAMPLE_COLUMNS).difference(source_dataset.schema.names)
    if missing:
        raise ValueError(f"Source Parquet is missing columns: {sorted(missing)}")

    upc_type = source_dataset.schema.field("upc").type
    arrow_upcs = coerce_values_for_arrow(selected_upcs, upc_type)
    sample = source_dataset.to_table(
        columns=SELECTED_SAMPLE_COLUMNS,
        filter=ds.field("upc").isin(arrow_upcs),
    ).to_pandas()
    for identifier in ["store", "upc", "store_upc"]:
        sample[identifier] = normalize_identifier(sample[identifier])
    return (
        sample.loc[
            sample["store"].isin(selected_stores)
            & sample["store_upc"].isin(eligible_panels)
        ]
        .sort_values(["store_upc", "week"], kind="mergesort")
        .reset_index(drop=True)
    )


def _write_selected_sample_cache(
    sample: pd.DataFrame,
    cache_path: Path,
    cache_manifest_path: Path,
    fingerprint: str,
) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(
        cache_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    manifest = {
        "fingerprint": fingerprint,
        "rows": int(len(sample)),
        "products": int(sample["upc"].nunique()),
        "stores": int(sample["store"].nunique()),
        "panels": int(sample["store_upc"].nunique()),
    }
    cache_manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _normalize_selected_sample(sample: pd.DataFrame) -> pd.DataFrame:
    sample = sample.copy()
    for identifier in ["store", "upc", "store_upc"]:
        sample[identifier] = normalize_identifier(sample[identifier])
    for column in [
        "week",
        "move",
        "discount_depth",
        "model_unit_price",
        "regular_price",
        "gross_margin_pct_observed",
    ]:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample["week"] = sample["week"].astype(int)
    sample["week_start"] = pd.to_datetime(sample["week_start"], errors="coerce")
    sample["week_end"] = pd.to_datetime(sample["week_end"], errors="coerce")
    sample["price_imputed"] = sample["price_imputed"].fillna(False).astype(bool)

    pricing_state = (
        sample["pricing_state"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    sample["promotion_indicator"] = pricing_state.isin(
        ["promotion", "promo"]
    ).astype("int8")
    sample["post_promotion_indicator"] = pricing_state.isin(
        ["post_promotion", "postpromo", "post_promo"]
    ).astype("int8")
    if sample.empty:
        raise ValueError("The compact selected sample is empty.")
    return sample


def load_or_build_selected_sample(
    source_data_path: Path,
    cache_path: Path,
    cache_manifest_path: Path,
    selected_upcs: list[str],
    selected_stores: list[str],
    eligible_panels: set[str],
    *,
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the compact selected sample or rebuild it from canonical selections."""
    fingerprint = selection_fingerprint(
        source_data_path,
        selected_upcs,
        selected_stores,
        eligible_panels,
    )
    manifest: dict[str, Any] = {}
    cache_current = False
    if use_cache and cache_path.exists() and cache_manifest_path.exists():
        manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_current = manifest.get("fingerprint") == fingerprint

    if cache_current and not force_rebuild:
        sample = pd.read_parquet(cache_path)
    else:
        sample = _read_selected_sample(
            source_data_path,
            selected_upcs,
            selected_stores,
            eligible_panels,
        )
        manifest = _write_selected_sample_cache(
            sample,
            cache_path,
            cache_manifest_path,
            fingerprint,
        )
    return _normalize_selected_sample(sample), manifest


def _complete_event_window(
    sample_indexed: pd.DataFrame,
    event: pd.Series,
    relative_weeks: np.ndarray,
    *,
    pre_event_weeks: int,
    require_no_other_promotion_in_window: bool,
    minimum_pre_period_mean_sales: float,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    panel = event["store_upc"]
    event_week = int(event["start_week"])
    desired_weeks = event_week + relative_weeks
    try:
        window = sample_indexed.loc[(panel, desired_weeks), :].reset_index()
    except KeyError:
        return None
    if len(window) != len(relative_weeks):
        return None

    window = window.sort_values("week").copy()
    window["relative_week"] = window["week"] - event_week
    if not np.array_equal(window["relative_week"].to_numpy(), relative_weeks):
        return None
    if require_no_other_promotion_in_window:
        other_promotion = window["promotion_indicator"].eq(1) & window[
            "relative_week"
        ].ne(0)
        if other_promotion.any():
            return None

    pre_period = window.loc[
        window["relative_week"].between(-pre_event_weeks, -1),
        "move",
    ]
    if len(pre_period) != pre_event_weeks:
        return None
    baseline = pre_period.mean()
    if not np.isfinite(baseline) or baseline <= minimum_pre_period_mean_sales:
        return None

    window["raw_baseline_sales"] = baseline
    window["raw_sales_index"] = 100.0 * window["move"] / baseline
    metadata = {
        "store_upc": panel,
        "store": event["store"],
        "upc": event["upc"],
        "event_week": event_week,
        "event_date": event["start_date"],
        "promotion_depth": event["median_depth"],
        "split": window.loc[window["relative_week"].eq(0), "split"].iloc[0],
    }
    return window, metadata


def _construct_isolated_event_windows(
    selected_sample: pd.DataFrame,
    promotion_episodes: pd.DataFrame,
    relative_weeks: np.ndarray,
    *,
    pre_event_weeks: int,
    require_single_week_promotion: bool,
    require_no_other_promotion_in_window: bool,
    minimum_pre_period_mean_sales: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_indexed = selected_sample.set_index(["store_upc", "week"]).sort_index()
    candidates = promotion_episodes.loc[
        promotion_episodes["consecutive_episode"]
    ].copy()
    if require_single_week_promotion:
        candidates = candidates.loc[candidates["promotion_weeks"].eq(1)]

    windows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for event_id, event in candidates.reset_index(drop=True).iterrows():
        constructed = _complete_event_window(
            sample_indexed,
            event,
            relative_weeks,
            pre_event_weeks=pre_event_weeks,
            require_no_other_promotion_in_window=(
                require_no_other_promotion_in_window
            ),
            minimum_pre_period_mean_sales=minimum_pre_period_mean_sales,
        )
        if constructed is None:
            continue
        window, metadata = constructed
        window["event_id"] = event_id
        metadata["event_id"] = event_id
        windows.append(window[EVENT_WINDOW_COLUMNS])
        metadata_rows.append(metadata)

    if not windows:
        raise RuntimeError(
            "No complete isolated promotion windows were found. "
            "Relax the event-window settings only after checking the data."
        )
    return pd.concat(windows, ignore_index=True), pd.DataFrame(metadata_rows)


def _load_prediction_inputs(
    rolling_manifest_path: Path,
    prediction_context_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_pickle(rolling_manifest_path)
    context = pd.read_pickle(prediction_context_path)
    predictions = pd.read_pickle(
        prediction_context_path.parent / "demand_predictions.pkl"
    )
    selected_model = str(manifest["model"].iloc[0])
    baseline = predictions.loc[
        predictions["model"].astype(str).eq(selected_model),
        ["store_upc", "upc", "week", "mu_hat_regular_counterfactual"],
    ].copy()
    context = context.merge(
        baseline,
        on=["store_upc", "upc", "week"],
        how="left",
        validate="one_to_one",
    )
    for identifier in ["store_upc", "upc"]:
        context[identifier] = normalize_identifier(context[identifier])
    context["week"] = pd.to_numeric(context["week"], errors="raise").astype(int)
    missing = set(PREDICTION_CONTEXT_COLUMNS).difference(context.columns)
    if missing:
        raise ValueError(f"Prediction context is missing columns: {sorted(missing)}")
    return manifest, context


def _attach_rolling_model_blocks(
    raw_event_windows: pd.DataFrame,
    isolated_events: pd.DataFrame,
    manifest: pd.DataFrame,
    prediction_context: pd.DataFrame,
    expected_window_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_windows = raw_event_windows.merge(
        prediction_context[PREDICTION_CONTEXT_COLUMNS],
        on=["store_upc", "upc", "week"],
        how="inner",
        validate="many_to_one",
    )
    event_windows["model_block"] = pd.Series(
        pd.NA,
        index=event_windows.index,
        dtype="Int64",
    )
    for row in manifest.itertuples(index=False):
        mask = event_windows["week"].between(
            int(row.first_forecast_week),
            int(row.last_forecast_week),
        )
        event_windows.loc[mask, "model_block"] = int(row.block_number)

    event_windows = event_windows.dropna(subset=["model_block"]).copy()
    event_windows["model_block"] = event_windows["model_block"].astype(int)
    completeness = event_windows.groupby("event_id", observed=True).agg(
        rows=("relative_week", "size"),
        distinct_weeks=("relative_week", "nunique"),
    )
    complete_ids = completeness.index[
        completeness["rows"].eq(expected_window_length)
        & completeness["distinct_weeks"].eq(expected_window_length)
    ]
    event_windows = event_windows.loc[
        event_windows["event_id"].isin(complete_ids)
    ].copy()
    isolated_events = isolated_events.loc[
        isolated_events["event_id"].isin(complete_ids)
    ].copy()
    if event_windows.empty:
        raise RuntimeError(
            "No isolated promotion events have complete rolling-PPML context. "
            "Check the rolling-model and prediction-context artifacts."
        )
    return event_windows, isolated_events


def _counterfactual_regular_price_frame(
    block_rows: pd.DataFrame,
    manifest_row: pd.Series,
) -> pd.DataFrame:
    counterfactual = block_rows.copy()
    counterfactual["promotion_indicator"] = 0
    counterfactual["post_promotion_indicator"] = 0
    counterfactual["discount_depth_model"] = 0.0
    counterfactual["discount_depth_sq"] = 0.0
    baseline_price = counterfactual["regular_price"].where(
        counterfactual["regular_price"].gt(0),
        counterfactual["model_unit_price"],
    )
    if baseline_price.isna().any() or ~baseline_price.gt(0).all():
        raise ValueError("Invalid prices in a rolling-model counterfactual block.")
    counterfactual["log_price_model"] = np.log(baseline_price)
    counterfactual["log_price_spline"] = counterfactual[
        "log_price_model"
    ].clip(
        float(manifest_row["spline_lower"]),
        float(manifest_row["spline_upper"]),
    )
    for column in ["store_upc", "upc", "calendar_month"]:
        counterfactual[column] = counterfactual[column].astype("string")
    return counterfactual


def _predict_no_promotion_baseline(
    event_windows: pd.DataFrame,
    manifest: pd.DataFrame,
    rolling_model_dir: Path,
) -> pd.DataFrame:
    event_windows = event_windows.copy()
    event_windows["baseline_mu_hat"] = np.nan
    event_windows["baseline_mu_hat"] = event_windows[
        "mu_hat_regular_counterfactual"
    ]

    if event_windows["baseline_mu_hat"].isna().any():
        raise RuntimeError("Some event rows lack no-promotion baseline predictions.")
    event_windows["residualized_sales_index"] = (
        100.0 * event_windows["move"] / event_windows["baseline_mu_hat"]
    )
    return event_windows


def _event_matrices(
    event_windows: pd.DataFrame,
    relative_weeks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "event_id",
        "store_upc",
        "relative_week",
        "move",
        "baseline_mu_hat",
    }
    missing = required.difference(event_windows.columns)
    if missing:
        raise RuntimeError(f"Event data are missing columns: {sorted(missing)}")

    panel_lookup = (
        event_windows[["event_id", "store_upc"]]
        .drop_duplicates()
        .set_index("event_id")["store_upc"]
    )
    observed = (
        event_windows.pivot(
            index="event_id",
            columns="relative_week",
            values="move",
        )
        .reindex(columns=relative_weeks)
        .sort_index()
    )
    baseline = (
        event_windows.pivot(
            index="event_id",
            columns="relative_week",
            values="baseline_mu_hat",
        )
        .reindex(columns=relative_weeks)
        .sort_index()
    )
    if observed.isna().any().any() or baseline.isna().any().any():
        raise RuntimeError("Observed or baseline event matrix is incomplete.")
    panel_lookup = panel_lookup.reindex(observed.index)
    if panel_lookup.isna().any():
        raise RuntimeError("Some event IDs lack store-product panel identifiers.")
    return (
        observed.to_numpy(dtype=float),
        baseline.to_numpy(dtype=float),
        panel_lookup.to_numpy(),
    )


def _cluster_bootstrap_event_path(
    observed: np.ndarray,
    baseline: np.ndarray,
    panel_labels: np.ndarray,
    relative_weeks: np.ndarray,
    *,
    bootstrap_replications: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    point_estimate = 100.0 * observed.sum(axis=0) / baseline.sum(axis=0)
    unique_panels = pd.unique(panel_labels)
    panel_positions = {
        panel: np.flatnonzero(panel_labels == panel) for panel in unique_panels
    }
    rng = np.random.default_rng(bootstrap_seed)
    estimates = np.empty((bootstrap_replications, len(relative_weeks)), dtype=float)
    for replication in range(bootstrap_replications):
        sampled_panels = rng.choice(
            unique_panels,
            size=len(unique_panels),
            replace=True,
        )
        positions = np.concatenate(
            [panel_positions[panel] for panel in sampled_panels]
        )
        estimates[replication] = (
            100.0 * observed[positions].sum(axis=0) / baseline[positions].sum(axis=0)
        )
    return pd.DataFrame(
        {
            "relative_week": relative_weeks,
            "sales_index": point_estimate,
            "ci_lower": np.quantile(estimates, 0.025, axis=0),
            "ci_upper": np.quantile(estimates, 0.975, axis=0),
            "n_events": observed.shape[0],
            "n_panels": len(unique_panels),
        }
    )


def build_residualized_event_paths(
    selected_sample: pd.DataFrame,
    promotion_episodes: pd.DataFrame,
    rolling_manifest_path: Path,
    prediction_context_path: Path,
    rolling_model_dir: Path,
    *,
    pre_event_weeks: int = 4,
    post_event_weeks: int = 6,
    require_single_week_promotion: bool = True,
    require_no_other_promotion_in_window: bool = True,
    minimum_pre_period_mean_sales: float = 1e-8,
    bootstrap_replications: int = 2_000,
    bootstrap_seed: int = 20260723,
) -> dict[str, Any]:
    """Construct isolated event windows and a rolling-PPML residualized path."""
    relative_weeks = np.arange(-pre_event_weeks, post_event_weeks + 1)
    raw_windows, isolated_events = _construct_isolated_event_windows(
        selected_sample,
        promotion_episodes,
        relative_weeks,
        pre_event_weeks=pre_event_weeks,
        require_single_week_promotion=require_single_week_promotion,
        require_no_other_promotion_in_window=require_no_other_promotion_in_window,
        minimum_pre_period_mean_sales=minimum_pre_period_mean_sales,
    )
    manifest, context = _load_prediction_inputs(
        rolling_manifest_path,
        prediction_context_path,
    )
    event_windows, isolated_events = _attach_rolling_model_blocks(
        raw_windows,
        isolated_events,
        manifest,
        context,
        expected_window_length=len(relative_weeks),
    )
    event_windows = _predict_no_promotion_baseline(
        event_windows,
        manifest,
        rolling_model_dir,
    )
    raw_event_path = (
        event_windows.groupby("relative_week", observed=True)
        .agg(
            mean_sales_index=("raw_sales_index", "mean"),
            n_events=("event_id", "nunique"),
        )
        .reset_index()
    )
    observed, baseline, panel_labels = _event_matrices(
        event_windows,
        relative_weeks,
    )
    residualized_event_path = _cluster_bootstrap_event_path(
        observed,
        baseline,
        panel_labels,
        relative_weeks,
        bootstrap_replications=bootstrap_replications,
        bootstrap_seed=bootstrap_seed,
    )
    print(
        "Complete isolated events with rolling-PPML baselines:",
        f"{observed.shape[0]:,}",
    )
    print("Store-product panels represented:", f"{pd.unique(panel_labels).size:,}")
    return {
        "relative_weeks": relative_weeks,
        "isolated_events": isolated_events,
        "event_windows": event_windows,
        "raw_event_path": raw_event_path,
        "residualized_event_path": residualized_event_path,
    }


__all__ = [
    "build_residualized_event_paths",
    "coerce_values_for_arrow",
    "format_depths",
    "load_or_build_selected_sample",
    "normalize_identifier",
    "parse_boolean",
    "selection_fingerprint",
]
