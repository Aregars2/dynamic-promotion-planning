"""Typed configuration for the empirical analysis."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping
import tomllib

from .paths import find_project_root


@dataclass(frozen=True)
class SampleConfig:
    excluded_weeks: tuple[int, ...] = (219,)
    non_cereal_upcs: tuple[int, ...] = (317,)
    discount_threshold: float = 0.05
    regular_price_window: str = "91D"
    regular_price_min_periods: int = 4
    regular_price_quantile: float = 0.90
    sales_outlier_min_move: int = 1_000
    sales_outlier_min_stores: int = 20
    sales_outlier_min_p95_ratio: float = 10.0
    parquet_compression: str = "zstd"
    random_seed: int = 42


@dataclass(frozen=True)
class DemandConfig:
    n_products: int = 8
    train_share: float = 0.60
    calibration_share: float = 0.20
    min_product_train_weeks: int = 100
    min_product_calibration_weeks: int = 40
    min_product_test_weeks: int = 40
    min_product_train_stores: int = 10
    min_product_distinct_prices: int = 8
    min_product_promotion_rows: int = 30
    min_product_regular_rows: int = 100
    max_product_imputed_share: float = 0.25
    min_product_relative_price_range: float = 0.10
    min_store_train_weeks: int = 100
    min_store_calibration_weeks: int = 40
    min_store_test_weeks: int = 40
    min_product_coverage_share: float = 0.75
    min_panel_train_weeks: int = 100
    min_panel_calibration_weeks: int = 30
    min_panel_test_weeks: int = 30
    refit_every_n_weeks: int = 4
    discount_depth_is_percent: bool = False


@dataclass(frozen=True)
class BehavioralConfig:
    primary_model: str = "product_promotion"
    random_seed: int = 42
    bootstrap_replications: int = 1_000
    event_pre_weeks: int = 4
    event_post_weeks: int = 6
    pre_event_weeks: tuple[int, ...] = (-3, -2, -1)
    post_lags_for_calibration: tuple[int, ...] = (1, 2, 3, 4)
    minimum_calibration_discount: float = 0.05
    minimum_isolated_events: int = 40
    minimum_depth_events: int = 80
    minimum_common_panels: int = 4
    holdout_pre_weeks: int = 3
    holdout_post_weeks: int = 4
    event_study_interval: tuple[float, float] = (0.05, 0.95)
    elasticity_bounds: tuple[float, float] = (0.10, 5.00)
    promotion_lift_bounds: tuple[float, float] = (-1.00, 3.00)
    displacement_bounds: tuple[float, float] = (0.0, 3.0)
    persistence_bounds: tuple[float, float] = (0.05, 0.95)
    minimum_persistence_signal: float = 0.05
    maximum_persistence_slope_ratio: float = 1.25
    persistence_partial_identification_grid: tuple[float, ...] = (0.15, 0.35, 0.60)
    gross_margin_column: str = "gross_margin_pct_observed"
    minimum_valid_margin: float = -1.0
    maximum_valid_margin: float = 0.95


@dataclass(frozen=True)
class PolicyConfig:
    decision_horizon: int = 12
    washout_horizons: tuple[int, ...] = (36,)
    cooldown_weeks: int = 2
    discount_factor: float = 1.0
    reimbursement_grid_start: float = 0.00
    reimbursement_grid_stop: float = 1.00
    reimbursement_grid_step: float = 0.05
    weekly_capacities: tuple[int, ...] = (1, 2, 3, 8)
    economic_profile: str = "weekly_predicted_demand_fixed_product_price_cost"
    forecast_information_mode: str = "ex_ante_common_origin"
    initial_conditions: str = "fresh_start"


@dataclass(frozen=True)
class SupportConfig:
    bin_width: float = 0.05
    minimum_depth: float = 0.05
    maximum_depth: float = 0.60
    minimum_observations: int = 15
    minimum_panels: int = 3
    maximum_positive_actions: int = 3


@dataclass(frozen=True)
class WashoutSelectionConfig:
    """Legacy diagnostic tolerances; no longer a user-facing design choice."""

    vdo_stability_tolerance: float = 0.05
    terminal_state_tolerance: float = 0.01


@dataclass(frozen=True)
class BoundaryConfig:
    reimbursement_start: float = 0.00
    reimbursement_stop: float = 1.00
    reimbursement_step: float = 0.01


@dataclass(frozen=True)
class RegularPriceSensitivityConfig:
    """Alternative high-quantile regular-price definitions for robustness."""

    quantiles: tuple[float, ...] = (0.80, 0.90, 0.95)


@dataclass(frozen=True)
class SensitivityConfig:
    regular_price: RegularPriceSensitivityConfig = RegularPriceSensitivityConfig()


@dataclass(frozen=True)
class AnalysisConfig:
    sample: SampleConfig = SampleConfig()
    demand: DemandConfig = DemandConfig()
    behavioral: BehavioralConfig = BehavioralConfig()
    policy: PolicyConfig = PolicyConfig()
    support: SupportConfig = SupportConfig()
    washout_selection: WashoutSelectionConfig = WashoutSelectionConfig()
    boundary_validation: BoundaryConfig = BoundaryConfig()
    sensitivity: SensitivityConfig = SensitivityConfig()


def _coerce_value(value: Any, default: Any) -> Any:
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _construct(cls: type, raw: Mapping[str, Any] | None):
    raw = dict(raw or {})
    defaults = cls()
    values = {}
    for field in fields(cls):
        default = getattr(defaults, field.name)
        values[field.name] = _coerce_value(raw.get(field.name, default), default)
    return cls(**values)


def load_analysis_config(path: Path | None = None) -> AnalysisConfig:
    """Load the canonical TOML configuration, using typed defaults for omissions."""
    root = find_project_root(path.parent if path is not None else None)
    config_path = path or root / "config" / "analysis.toml"
    if not config_path.is_file():
        return AnalysisConfig()

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    return AnalysisConfig(
        sample=_construct(SampleConfig, raw.get("sample")),
        demand=_construct(DemandConfig, raw.get("demand")),
        behavioral=_construct(BehavioralConfig, raw.get("behavioral")),
        policy=_construct(PolicyConfig, raw.get("policy")),
        support=_construct(SupportConfig, raw.get("support")),
        washout_selection=_construct(
            WashoutSelectionConfig,
            raw.get("washout_selection"),
        ),
        boundary_validation=_construct(
            BoundaryConfig,
            raw.get("boundary_validation"),
        ),
        sensitivity=SensitivityConfig(
            regular_price=_construct(
                RegularPriceSensitivityConfig,
                (raw.get("sensitivity") or {}).get("regular_price"),
            ),
        ),
    )
