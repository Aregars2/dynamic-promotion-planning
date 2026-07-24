"""Construction of product-specific empirically supported promotion actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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

__all__ = ['SupportedActionConfig', 'build_product_supported_action_sets']
