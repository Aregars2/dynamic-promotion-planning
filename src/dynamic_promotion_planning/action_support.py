"""Construction of product-specific empirically supported promotion actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupportedActionConfig:
    """Rules used to classify and select empirically supported discounts."""

    bin_width: float = 0.05
    minimum_depth: float = 0.05
    maximum_depth: float = 0.60
    minimum_observations: int = 15
    minimum_panels: int = 3
    maximum_positive_actions: int = 3
    center_rounding: int = 2


def _prepare_calibration_rows(
    history: pd.DataFrame,
    config: SupportedActionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = history.loc[history["sample_period"].eq("calibration")].copy()
    calibration["upc"] = calibration["upc"].astype(str)

    promotion_rows = calibration.loc[
        calibration["promotion_indicator"].eq(1)
        & calibration["discount_depth_model"].between(
            config.minimum_depth,
            config.maximum_depth,
            inclusive="both",
        )
    ].copy()
    promotion_rows["support_bin_center"] = (
        np.round(promotion_rows["discount_depth_model"] / config.bin_width)
        * config.bin_width
    ).clip(config.minimum_depth, config.maximum_depth)
    return calibration, promotion_rows


def _reference_action_row(
    upc: str,
    product_name: str,
    regular_rows: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "upc": upc,
        "product_name": product_name,
        "action": 0.0,
        "bin_center": 0.0,
        "support_count": int(len(regular_rows)),
        "support_panels": int(regular_rows["store_upc"].nunique()),
        "support_stores": int(regular_rows["store"].nunique()),
        "depth_q10": 0.0,
        "depth_q50": 0.0,
        "depth_q90": 0.0,
        "supported": True,
        "selected_for_grid": True,
        "selection_reason": "reference_action",
    }


def _summarize_promotion_clusters(
    product_promotions: pd.DataFrame,
    *,
    upc: str,
    product_name: str,
    config: SupportedActionConfig,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for bin_center, cluster in product_promotions.groupby(
        "support_bin_center",
        observed=True,
    ):
        support_count = int(len(cluster))
        support_panels = int(cluster["store_upc"].nunique())
        support_stores = int(cluster["store"].nunique())
        supported = (
            support_count >= config.minimum_observations
            and support_panels >= config.minimum_panels
        )
        clusters.append(
            {
                "upc": upc,
                "product_name": product_name,
                "action": float(np.round(bin_center, config.center_rounding)),
                "bin_center": float(np.round(bin_center, config.center_rounding)),
                "support_count": support_count,
                "support_panels": support_panels,
                "support_stores": support_stores,
                "depth_q10": float(
                    cluster["discount_depth_model"].quantile(0.10)
                ),
                "depth_q50": float(cluster["discount_depth_model"].median()),
                "depth_q90": float(
                    cluster["discount_depth_model"].quantile(0.90)
                ),
                "supported": bool(supported),
                "selected_for_grid": False,
                "selection_reason": "below_support_threshold",
            }
        )
    return clusters


def _select_top_supported_cluster_indices(
    clusters: list[dict[str, Any]],
    maximum_positive_actions: int,
) -> list[int]:
    supported_indices = [
        index for index, row in enumerate(clusters) if row["supported"]
    ]
    if not supported_indices:
        return []

    return sorted(
        supported_indices,
        key=lambda index: (
            -int(clusters[index]["support_count"]),
            -int(clusters[index]["support_panels"]),
            float(clusters[index]["bin_center"]),
        ),
    )[:maximum_positive_actions]


def _mark_selected_clusters(
    clusters: list[dict[str, Any]],
    selected_indices: list[int],
) -> tuple[list[dict[str, Any]], tuple[float, ...]]:
    selected_set = set(selected_indices)
    for index, row in enumerate(clusters):
        if index in selected_set:
            row["selected_for_grid"] = True
            row["selection_reason"] = "top_supported_bin"

    selected_actions = sorted(
        {float(clusters[index]["action"]) for index in selected_indices}
    )
    return clusters, tuple([0.0, *selected_actions])


def build_product_supported_action_sets(
    history: pd.DataFrame,
    selected_products: list[str],
    product_names: dict[str, str],
    config: SupportedActionConfig,
) -> tuple[pd.DataFrame, dict[str, tuple[float, ...]]]:
    """Construct auditable support summaries and product-specific action grids."""
    calibration, promotion_rows = _prepare_calibration_rows(history, config)
    support_rows: list[dict[str, Any]] = []
    action_sets: dict[str, tuple[float, ...]] = {}

    for raw_upc in selected_products:
        upc = str(raw_upc)
        product_name = product_names.get(upc, upc)
        product_rows = calibration.loc[calibration["upc"].eq(upc)]
        product_promotions = promotion_rows.loc[promotion_rows["upc"].eq(upc)]
        regular_rows = product_rows.loc[product_rows["promotion_indicator"].eq(0)]

        support_rows.append(_reference_action_row(upc, product_name, regular_rows))
        clusters = _summarize_promotion_clusters(
            product_promotions,
            upc=upc,
            product_name=product_name,
            config=config,
        )
        selected_indices = _select_top_supported_cluster_indices(
            clusters,
            config.maximum_positive_actions,
        )
        clusters, action_set = _mark_selected_clusters(clusters, selected_indices)
        support_rows.extend(clusters)
        action_sets[upc] = action_set

    return pd.DataFrame(support_rows), action_sets


def build_supported_action_sets_from_table(
    support_table: pd.DataFrame,
    products: list[str],
    *,
    minimum_depth: float,
    maximum_depth: float,
    minimum_observations: int,
    minimum_panels: int,
    maximum_positive_actions: int,
) -> dict[str, tuple[float, ...]]:
    """Select canonical 5pp support-bin actions from an exported support table."""
    frame = support_table.copy()
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    aliases = {
        "bin_center": "action",
        "depth_cluster": "action",
        "support_count": "observations",
        "support_panels": "panels",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    required = ["upc", "action", "observations", "panels"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Support table is missing columns: {missing}")
    for column in ["action", "observations", "panels"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["upc"] = frame["upc"].astype(str)
    frame = frame.dropna(subset=required).loc[
        lambda data: data["action"].between(minimum_depth, maximum_depth)
        & data["observations"].ge(minimum_observations)
        & data["panels"].ge(minimum_panels)
    ].copy()
    frame = frame.sort_values(
        ["upc", "observations", "panels", "action"],
        ascending=[True, False, False, True],
    ).groupby("upc", observed=True, sort=False).head(maximum_positive_actions)
    return {
        str(upc): tuple([0.0, *sorted(group["action"].astype(float).unique())])
        for upc in map(str, products)
        for group in [frame.loc[frame["upc"].eq(str(upc))]]
    }


__all__ = [
    "SupportedActionConfig",
    "build_product_supported_action_sets",
    "build_supported_action_sets_from_table",
]
