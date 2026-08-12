from __future__ import annotations

import numpy as np
import pandas as pd

from dynamic_promotion_planning.demand import build_common_origin_ex_ante_frame


def _history() -> pd.DataFrame:
    rows = []
    for panel, price in [("s1_u", 2.0), ("s2_u", 3.0)]:
        for week in [1, 2, 3, 4]:
            rows.append({
                "store": panel[:2], "upc": "u", "store_upc": panel, "week": week,
                "move": 20 + week, "regular_price": price, "model_unit_price": price,
                "price_imputed_indicator": 0, "calendar_month": "jan",
                "scaled_time_trend": week / 10, "thanksgiving_week": 0,
                "christmas_week": 0, "new_year_week": 0, "easter_week": 0,
                "promotion_indicator": 0, "post_promotion_indicator": 0,
            })
    return pd.DataFrame(rows)


def test_common_origin_grid_is_complete_and_uses_origin_known_panel_inputs():
    history = _history()
    future = history.loc[history.week.eq(4)].copy()
    future = pd.concat([future.assign(week=5), future.assign(week=6)], ignore_index=True)
    # These values deliberately differ from origin values and must never be used.
    future["move"] = 9_999
    future["regular_price"] = 99.0
    future["promotion_indicator"] = 1
    output = build_common_origin_ex_ante_frame(
        pd.concat([history, future], ignore_index=True), np.array([5, 6]),
        planning_origin_week=5, calendar_frame=future,
    )
    assert len(output) == 4
    assert output.groupby("week", observed=True)["store_upc"].nunique().eq(2).all()
    assert set(output["regular_price"]) == {2.0, 3.0}
    assert output["promotion_indicator"].eq(0).all()
    assert output["post_promotion_indicator"].eq(0).all()
    assert output["move"].isna().all()


def test_future_sales_prices_and_promotions_cannot_change_ex_ante_grid():
    history = _history()
    future = pd.concat([
        history.loc[history.week.eq(4)].assign(week=5),
        history.loc[history.week.eq(4)].assign(week=6),
    ], ignore_index=True)
    source = pd.concat([history, future], ignore_index=True)
    original = build_common_origin_ex_ante_frame(source, np.array([5, 6]), planning_origin_week=5, calendar_frame=future)
    perturbed = source.copy()
    changed = perturbed.week.ge(5)
    perturbed.loc[changed, ["move", "regular_price", "model_unit_price"]] = [12345, 77, 77]
    perturbed.loc[changed, "promotion_indicator"] = 1
    candidate = build_common_origin_ex_ante_frame(perturbed, np.array([5, 6]), planning_origin_week=5, calendar_frame=future)
    columns = ["store_upc", "week", "regular_price", "model_unit_price", "promotion_indicator", "post_promotion_indicator"]
    pd.testing.assert_frame_equal(original[columns].sort_values(columns[:2]).reset_index(drop=True), candidate[columns].sort_values(columns[:2]).reset_index(drop=True))
