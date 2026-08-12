from __future__ import annotations

import numpy as np
import pandas as pd

from dynamic_promotion_planning.demand import (
    _fit_ppml,
    _prediction_block,
    regular_counterfactual_audit,
)


def _frame() -> pd.DataFrame:
    rows = []
    for panel, scale in [("a", 10.0), ("b", 16.0)]:
        for week in range(1, 7):
            price = 2.0 if week != 6 else 1.6
            rows.append({
                "store": panel,
                "upc": "u",
                "store_upc": panel,
                "week": week,
                "week_start": pd.Timestamp("2020-01-01"),
                "move": scale * (1.25 if price < 2 else 1.0),
                "model_unit_price": price,
                "unit_price_observed": price,
                "regular_price": 2.0,
                "discount_depth_model": 0.2 if week == 6 else 0.0,
                "price_imputed": False,
                "pricing_state": "promotion" if week == 6 else "regular",
                "promotion_indicator": int(week == 6),
                "post_promotion_indicator": 0,
                "reference_price": 2.0,
                "log_price_model": np.log(price),
            })
    return pd.DataFrame(rows)


def test_in_memory_ppml_counterfactual_matches_exported_block_and_store_sum():
    frame = _frame()
    fit = frame.loc[frame.week < 6].copy()
    prediction = frame.loc[frame.week.eq(6)].copy()
    formula = "move ~ 0 + C(store_upc) + log_price_model + promotion_indicator"
    result = _fit_ppml(fit, formula, "test", np.array([6]))
    block = _prediction_block(
        result, prediction, model_name="test", block_number=0,
        fit_last_week=5, lower=np.log(1.0), upper=np.log(3.0),
    )
    cf_input = prediction.copy()
    cf_input["model_unit_price"] = cf_input["regular_price"]
    cf_input["log_price_model"] = np.log(cf_input["regular_price"])
    cf_input["promotion_indicator"] = 0
    cf_input["post_promotion_indicator"] = 0
    expected = np.clip(result.predict(cf_input), 1e-12, None)
    np.testing.assert_allclose(block["mu_hat_regular_counterfactual"], expected)
    baseline = block.groupby(["upc", "week"], observed=True)["mu_hat_regular_counterfactual"].sum()
    assert baseline.iloc[0] == block["mu_hat_regular_counterfactual"].sum()


def test_promotional_and_postpromotion_rows_are_retained_and_reset():
    predictions = pd.DataFrame({
        "model": ["m", "m", "m"], "split": ["test"] * 3,
        "store": ["s"] * 3, "upc": ["u"] * 3, "store_upc": ["s_u"] * 3,
        "week": [1, 2, 3], "promotion_indicator": [1, 0, 0],
        "post_promotion_indicator": [0, 1, 0], "model_unit_price": [1.5, 2.0, 2.0],
        "regular_price": [2.0] * 3, "counterfactual_model_unit_price": [2.0] * 3,
        "mu_hat": [1.0, 1.0, 1.0], "mu_hat_regular_counterfactual": [1.2, 1.1, 1.0],
    })
    audit = regular_counterfactual_audit(predictions, model_name="m")
    assert set(audit["week"]) == {1, 2}
    assert audit["counterfactual_model_unit_price"].eq(audit["regular_price"]).all()
    assert audit["counterfactual_promotion_indicator"].eq(0).all()
    assert audit["counterfactual_post_promotion_indicator"].eq(0).all()
