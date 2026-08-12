from __future__ import annotations

from dynamic_promotion_planning.config import AnalysisConfig, load_analysis_config
from dynamic_promotion_planning.policy import maximum_feasible_promotions


def test_default_configuration_is_typed() -> None:
    config = AnalysisConfig()
    assert config.policy.decision_horizon == 12
    assert config.support.minimum_observations == 15


def test_repository_configuration_loads() -> None:
    config = load_analysis_config()
    assert config.demand.n_products == 8
    assert config.boundary_validation.reimbursement_step == 0.01
    assert config.policy.discount_factor == 1.0
    assert config.policy.washout_horizons == (36,)
    assert config.support.minimum_depth == 0.05
    assert config.sensitivity.regular_price.quantiles == (0.80, 0.90, 0.95)


def test_promotion_cap_is_implied_by_horizon_and_cooldown() -> None:
    assert maximum_feasible_promotions(12, 2) == 4
    assert maximum_feasible_promotions(3, 1) == 2
