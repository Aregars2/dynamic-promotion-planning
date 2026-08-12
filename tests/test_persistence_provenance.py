from __future__ import annotations

import pandas as pd

from dynamic_promotion_planning.calibration import CalibrationConfig, _expand_persistence_draws, _shrink_product_draws


def test_pooled_fallback_draws_cannot_claim_product_depth_decay():
    config = CalibrationConfig(bootstrap_replications=2, persistence_grid=(0.2, 0.6))
    pooled = pd.DataFrame({"price_elasticity": [1.0, 1.1], "promotion_lift_log": [0.2, 0.3], "displacement_strength": [0.4, 0.5], "inventory_persistence": [0.4, 0.5]})
    base = _shrink_product_draws(pd.DataFrame(), pooled, reliability=0.0, config=config)
    expanded = _expand_persistence_draws(base, config)
    assert expanded["product_calibration_source"].eq("pooled_fallback").all()
    assert expanded["persistence_source"].eq("pooled_fallback_grid").all()
    assert not expanded["persistence_informative"].any()
