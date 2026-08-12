from __future__ import annotations

import numpy as np
import pandas as pd

from dynamic_promotion_planning.action_support import (
    SupportedActionConfig,
    build_product_supported_action_sets,
)
from dynamic_promotion_planning.promotion import canonical_promotion_indicator
from dynamic_promotion_planning.demand import poisson_deviance_contribution
from dynamic_promotion_planning.descriptives import parse_boolean


def test_supported_action_grid_contains_reference_and_supported_depth() -> None:
    history = pd.DataFrame(
        {
            "sample_period": ["calibration"] * 22,
            "upc": ["A"] * 22,
            "store": [f"S{i % 4}" for i in range(22)],
            "store_upc": [f"S{i % 4}_A" for i in range(22)],
            "promotion_indicator": [0, 0] + [1] * 20,
            "discount_depth_model": [0.0, 0.0] + [0.10] * 20,
        }
    )
    table, action_sets = build_product_supported_action_sets(
        history=history,
        selected_products=["A"],
        product_names={"A": "Product A"},
        config=SupportedActionConfig(
            minimum_observations=15,
            minimum_panels=3,
            maximum_positive_actions=1,
        ),
    )
    assert action_sets["A"] == (0.0, 0.1)
    selected = table.loc[table["selected_for_grid"]]
    assert set(selected["action"]) == {0.0, 0.1}


def test_canonical_promotion_rule_excludes_unrecorded_three_to_five_percent_discount() -> None:
    history = pd.DataFrame(
        {
            "discount_depth_model": [0.03, 0.049, 0.05, 0.04],
            "promo_recorded": [0, 0, 0, 1],
        }
    )
    assert canonical_promotion_indicator(
        history, recorded_column="promo_recorded"
    ).tolist() == [0, 0, 1, 1]


def test_action_support_uses_bin_centers_and_top_support_rule() -> None:
    rows = []
    for depth, count, panels in [
        (0.06, 20, 3),  # 5pp bin; must select action 0.05, not median 0.06.
        (0.11, 25, 4),
        (0.16, 25, 3),
        (0.21, 20, 5),
        (0.26, 20, 3),
    ]:
        for index in range(count):
            rows.append(
                {
                    "sample_period": "calibration",
                    "upc": "A",
                    "store": f"S{index % panels}",
                    "store_upc": f"S{index % panels}_A",
                    "promotion_indicator": 1,
                    "discount_depth_model": depth,
                }
            )
    table, action_sets = build_product_supported_action_sets(
        pd.DataFrame(rows),
        selected_products=["A"],
        product_names={"A": "Product A"},
        config=SupportedActionConfig(
            minimum_observations=15,
            minimum_panels=3,
            maximum_positive_actions=3,
        ),
    )
    assert action_sets["A"] == (0.0, 0.1, 0.15, 0.2)
    positive = [action for action in action_sets["A"] if action > 0]
    assert all(action >= 0.05 for action in positive)
    assert all(round(action / 0.05) * 0.05 == action for action in positive)
    selected = table.loc[table["selected_for_grid"] & table["action"].gt(0)]
    assert set(selected["selection_reason"]) == {"top_supported_bin"}


def test_poisson_deviance_is_zero_at_exact_prediction() -> None:
    observed = np.array([0.0, 1.0, 5.0])
    deviance = poisson_deviance_contribution(observed, observed)
    assert np.allclose(deviance, 0.0)


def test_parse_boolean_accepts_common_encodings() -> None:
    values = pd.Series(["true", "0", "yes", "False"])
    parsed = parse_boolean(values)
    assert parsed.tolist() == [True, False, True, False]
