"""Sample-construction helpers for the Dominick's cereal panel."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .config import SampleConfig


def fourth_thursday_of_november(year: int) -> pd.Timestamp:
    """Return the date of U.S. Thanksgiving for a given year."""
    november_first = pd.Timestamp(year=year, month=11, day=1)
    offset = (3 - november_first.weekday()) % 7
    first_thursday = november_first + pd.Timedelta(days=offset)
    return first_thursday + pd.Timedelta(weeks=3)


def _candidate_sales_rows(
    demand_data: pd.DataFrame,
    minimum_move: float,
) -> pd.DataFrame:
    candidates = demand_data.loc[
        demand_data["move"].ge(minimum_move),
        ["store", "upc", "week", "move"],
    ].copy()
    candidates["_row_index"] = candidates.index
    return candidates


def _leave_one_store_out_comparisons(
    demand_data: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    candidate_keys = candidates[["upc", "week"]].drop_duplicates()
    comparison_pool = demand_data[["store", "upc", "week", "move"]].merge(
        candidate_keys,
        on=["upc", "week"],
        how="inner",
        validate="many_to_many",
    )
    comparisons = candidates[
        ["_row_index", "store", "upc", "week", "move"]
    ].merge(
        comparison_pool,
        on=["upc", "week"],
        how="left",
        suffixes=("_candidate", "_comparison"),
        validate="many_to_many",
    )
    return comparisons.loc[
        comparisons["store_candidate"].ne(comparisons["store_comparison"])
    ]


def _summarize_comparison_distribution(comparisons: pd.DataFrame) -> pd.DataFrame:
    return (
        comparisons.groupby("_row_index", observed=True)
        .agg(
            comparison_stores=("store_comparison", "nunique"),
            comparison_median_move=("move_comparison", "median"),
            comparison_p95_move=(
                "move_comparison",
                lambda values: values.quantile(0.95),
            ),
        )
        .reset_index()
    )


def _classify_sales_outliers(
    candidates: pd.DataFrame,
    comparisons: pd.DataFrame,
    config: SampleConfig,
) -> pd.DataFrame:
    classified = candidates.merge(
        comparisons,
        on="_row_index",
        how="left",
        validate="one_to_one",
    )
    classified["reporting_stores"] = classified["comparison_stores"] + 1
    classified["move_to_p95"] = (
        classified["move"] / classified["comparison_p95_move"].clip(lower=1)
    )
    classified["move_to_median"] = (
        classified["move"] / classified["comparison_median_move"].clip(lower=1)
    )
    classified["sales_outlier"] = (
        classified["reporting_stores"].ge(config.sales_outlier_min_stores)
        & classified["move_to_p95"].ge(config.sales_outlier_min_p95_ratio)
    )
    return classified


def _sales_outlier_summary(annotated: pd.DataFrame) -> pd.Series:
    flagged = annotated["sales_outlier"]
    return pd.Series(
        {
            "flagged_rows": int(flagged.sum()),
            "flagged_share": float(flagged.mean()),
            "flagged_units": float(annotated.loc[flagged, "move"].sum()),
            "flagged_stores": int(annotated.loc[flagged, "store"].nunique()),
        },
        name="value",
    )


def _build_sales_outlier_audit(
    annotated: pd.DataFrame,
    classified: pd.DataFrame,
    product_lookup: pd.DataFrame,
) -> pd.DataFrame:
    flagged = classified.loc[classified["sales_outlier"]]
    audit = annotated.loc[annotated["sales_outlier"]].copy()
    audit["_row_index"] = audit.index
    comparison_columns = [
        "_row_index",
        "reporting_stores",
        "comparison_median_move",
        "comparison_p95_move",
        "move_to_median",
        "move_to_p95",
    ]
    return (
        audit.merge(
            product_lookup,
            on="upc",
            how="left",
            validate="many_to_one",
        )
        .merge(
            flagged[comparison_columns],
            on="_row_index",
            how="left",
            validate="one_to_one",
        )
        .sort_values("move_to_p95", ascending=False)
    )


def detect_cross_store_sales_outliers(
    demand_data: pd.DataFrame,
    product_lookup: pd.DataFrame,
    demand_columns: list[str],
    config: SampleConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Apply the pre-specified leave-one-store-out sales-anomaly rule."""
    config = config or SampleConfig()
    candidates = _candidate_sales_rows(
        demand_data,
        config.sales_outlier_min_move,
    )
    pairwise_comparisons = _leave_one_store_out_comparisons(
        demand_data,
        candidates,
    )
    comparison_summary = _summarize_comparison_distribution(pairwise_comparisons)
    classified = _classify_sales_outliers(
        candidates,
        comparison_summary,
        config,
    )

    annotated = demand_data.copy(deep=False)
    annotated["sales_outlier"] = False
    flagged_indices = classified.loc[
        classified["sales_outlier"],
        "_row_index",
    ].to_numpy()
    annotated.loc[flagged_indices, "sales_outlier"] = True

    summary = _sales_outlier_summary(annotated)
    audit = _build_sales_outlier_audit(annotated, classified, product_lookup)
    clean = annotated.loc[~annotated["sales_outlier"], demand_columns].copy(
        deep=False
    )

    if len(annotated) - len(clean) != int(summary["flagged_rows"]):
        raise AssertionError("Sales-anomaly accounting does not reconcile.")
    return clean, audit, summary


def write_parquet_outputs(
    outputs: Mapping[str, tuple[Path, pd.DataFrame]],
    compression: str = "zstd",
) -> pd.DataFrame:
    """Write and verify canonical Parquet outputs."""
    rows = []
    for label, (path, frame) in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(
            path,
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        written = pd.read_parquet(path)
        rows.append(
            {
                "output": label,
                "path": str(path),
                "rows_expected": int(len(frame)),
                "rows_written": int(len(written)),
                "columns_expected": int(frame.shape[1]),
                "columns_written": int(written.shape[1]),
            }
        )

    audit = pd.DataFrame(rows)
    verified = audit["rows_expected"].eq(audit["rows_written"]) & audit[
        "columns_expected"
    ].eq(audit["columns_written"])
    if not verified.all():
        raise AssertionError("One or more Parquet outputs failed verification.")
    return audit
