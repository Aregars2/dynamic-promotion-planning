from __future__ import annotations

import numpy as np
import pandas as pd

from dynamic_promotion_planning.calibration import (
    _apply_empirical_bayes_draws,
    _empirical_bayes_weights,
)


def _raw_draws(price: list[float], lift: list[float], displacement: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bootstrap_id": range(len(price)),
            "price_elasticity": price,
            "promotion_lift_log": lift,
            "displacement_strength": displacement,
        }
    )


def test_empirical_bayes_weights_are_parameter_specific_and_match_moments() -> None:
    raw = {
        "a": _raw_draws([1.0, 1.2, 0.8], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]),
        "b": _raw_draws([2.0, 2.2, 1.8], [0.3, 0.3, 0.3], [1.1, 1.1, 1.1]),
    }
    weights, detail, summary = _empirical_bayes_weights(raw)
    price = summary.set_index("parameter").loc["price_elasticity"]
    expected_within = np.mean([np.var([1.0, 1.2, 0.8], ddof=1), np.var([2.0, 2.2, 1.8], ddof=1)])
    expected_between = np.var([1.0, 2.0], ddof=1)
    assert np.isclose(price["mean_within_product_bootstrap_variance"], expected_within)
    assert np.isclose(price["tau2"], expected_between - expected_within)
    assert weights["a"]["price_elasticity"] != weights["a"]["promotion_lift_log"]
    assert detail.groupby("parameter")["upc"].nunique().eq(2).all()


def test_eb_mixes_aligned_bootstrap_parents_and_full_pools_at_zero_tau2() -> None:
    raw = {
        "a": _raw_draws([1.0, 1.2], [0.2, 0.2], [0.4, 0.4]),
        "b": _raw_draws([1.0, 1.2], [0.2, 0.2], [0.4, 0.4]),
    }
    weights, _, summary = _empirical_bayes_weights(raw)
    assert summary.set_index("parameter").loc["price_elasticity", "tau2_at_zero"]
    pooled = pd.DataFrame(
        {
            "price_elasticity": [3.0, 4.0],
            "promotion_lift_log": [0.5, 0.6],
            "displacement_strength": [0.7, 0.8],
            "inventory_persistence": [0.3, 0.3],
        }
    )
    mixed = _apply_empirical_bayes_draws(raw["a"], pooled, weights["a"])
    assert np.allclose(mixed["price_elasticity"], pooled["price_elasticity"])
    assert np.allclose(mixed["promotion_lift_log"], pooled["promotion_lift_log"])
    assert np.allclose(mixed["displacement_strength"], pooled["displacement_strength"])
