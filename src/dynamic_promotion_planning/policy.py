"""Core demand-displacement policy optimization routines.

The active notebooks use this module for schedule construction, dynamic and
myopic policy evaluation, terminal washout, and decomposition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack


@dataclass(frozen=True)
class PlanningSpec:
    """Configuration for the promotion-planning problem."""

    decision_horizon: int = 12
    washout_horizon: int = 12
    cooldown: int = 2
    max_promotions: int = 4
    discount_factor: float = 1.0
    reimbursement_min: float = 0.0
    reimbursement_max: float = 1.0
    # Deprecated aliases retained while legacy tests and artifacts are migrated.
    alpha_min: float | None = None
    alpha_max: float | None = None

    @property
    def evaluation_horizon(self) -> int:
        return self.decision_horizon + self.washout_horizon


def maximum_feasible_promotions(
    decision_horizon: int,
    cooldown: int,
) -> int:
    """Return the promotion cap implied by the horizon and cooldown."""
    if decision_horizon < 1:
        raise ValueError("decision_horizon must be positive.")
    if cooldown < 0:
        raise ValueError("cooldown must be nonnegative.")
    return (decision_horizon + cooldown) // (cooldown + 1)


@dataclass(frozen=True)
class SupportSpec:
    """Baseline empirical-support rule for positive promotion depths."""

    bin_width: float = 0.05
    minimum_depth: float = 0.05
    maximum_depth: float = 0.60
    minimum_observations: int = 15
    minimum_panels: int = 3
    maximum_positive_actions: int = 3


REQUIRED_DRAW_COLUMNS = (
    "upc",
    "price_elasticity",
    "promotion_lift_log",
    "displacement_strength",
    "inventory_persistence",
    "draw_weight",
    "base_demand",
    "regular_price",
    "unit_cost",
)


def normalize_identifier(values: pd.Series | Sequence[Any]) -> pd.Series:
    """Normalize UPC/store-like identifiers without scientific notation."""

    series = pd.Series(values, copy=False)
    numeric = pd.to_numeric(series, errors="coerce")
    result = series.astype("string").str.strip()
    integer_like = numeric.notna() & np.isclose(numeric, np.round(numeric))
    result.loc[integer_like] = (
        numeric.loc[integer_like].round().astype("Int64").astype("string")
    )
    return result.astype(str)


def _coerce_draw_frame(raw: Any) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, dict):
        for key in ("product_draws", "behavioral_draws", "parameter_draws", "draws"):
            value = raw.get(key)
            if isinstance(value, pd.DataFrame):
                return value.copy()
        if raw and all(isinstance(value, pd.DataFrame) for value in raw.values()):
            frames: list[pd.DataFrame] = []
            for key, frame in raw.items():
                frame = frame.copy()
                if "upc" not in frame.columns:
                    frame["upc"] = str(key)
                frames.append(frame)
            return pd.concat(frames, ignore_index=True)
    raise TypeError(
        "Behavioral draws must be a pandas DataFrame or a dictionary containing one."
    )


def prepare_behavioral_draws(raw: Any) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Validate and convert the behavioral-draw artifact to numeric arrays by product."""

    frame = _coerce_draw_frame(raw)
    frame.columns = frame.columns.astype(str).str.strip().str.lower()

    aliases = {
        "epsilon": "price_elasticity",
        "elasticity": "price_elasticity",
        "gamma": "promotion_lift_log",
        "promotion_lift": "promotion_lift_log",
        "psi": "displacement_strength",
        "persistence": "inventory_persistence",
        "weight": "draw_weight",
        "q_base": "base_demand",
        "price": "regular_price",
        "cost": "unit_cost",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]

    missing = [column for column in REQUIRED_DRAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Behavioral-draw artifact is missing columns: {missing}")

    frame["upc"] = normalize_identifier(frame["upc"])
    numeric_columns = [column for column in REQUIRED_DRAW_COLUMNS if column != "upc"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = np.ones(len(frame), dtype=bool)
    for column in numeric_columns:
        valid &= np.isfinite(frame[column].to_numpy(dtype=float))
    valid &= frame["draw_weight"].to_numpy(dtype=float) > 0
    valid &= frame["base_demand"].to_numpy(dtype=float) > 0
    valid &= frame["regular_price"].to_numpy(dtype=float) > 0
    valid &= frame["unit_cost"].to_numpy(dtype=float) >= 0
    valid &= frame["inventory_persistence"].between(0, 1, inclusive="both").to_numpy()

    frame = frame.loc[valid].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid behavioral draws remain after validation.")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for upc, group in frame.groupby("upc", observed=True, sort=True):
        weights = group["draw_weight"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        arrays[str(upc)] = {
            "epsilon": group["price_elasticity"].to_numpy(dtype=float),
            "gamma": group["promotion_lift_log"].to_numpy(dtype=float),
            "psi": np.clip(
                group["displacement_strength"].to_numpy(dtype=float), 0.0, None
            ),
            "r": np.clip(
                group["inventory_persistence"].to_numpy(dtype=float), 0.0, 1.0
            ),
            "weights": weights,
            "base_demand": group["base_demand"].to_numpy(dtype=float),
            "regular_price": group["regular_price"].to_numpy(dtype=float),
            "unit_cost": group["unit_cost"].to_numpy(dtype=float),
        }

    return frame, arrays


def prepare_support_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize the saved product-depth support table."""

    frame = raw.copy()
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    aliases = {
        "bin_center": "depth_cluster",
        "depth": "depth_cluster",
        "promotion_depth": "depth_cluster",
        "action_depth": "depth_cluster",
        "representative_depth": "depth_cluster",
        "depth_center": "depth_cluster",
        "support_count": "observations",
        "n_observations": "observations",
        "observation_count": "observations",
        "support_observations": "observations",
        "count": "observations",
        "n_panels": "panels",
        "panel_count": "panels",
        "support_panels": "panels",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]

    required = ["upc", "depth_cluster", "observations", "panels"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Support table is missing columns: {missing}")

    frame["upc"] = normalize_identifier(frame["upc"])
    for column in ["depth_cluster", "observations", "panels"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).copy()
    frame["depth_cluster"] = frame["depth_cluster"].round(6)
    frame["observations"] = frame["observations"].astype(int)
    frame["panels"] = frame["panels"].astype(int)
    return frame.sort_values(["upc", "depth_cluster"]).reset_index(drop=True)


def build_supported_action_sets(
    support_table: pd.DataFrame,
    products: Sequence[str],
    support_spec: SupportSpec,
) -> dict[str, tuple[float, ...]]:
    """Build nested product-specific action sets from a fixed master depth grid."""
    from .action_support import build_supported_action_sets_from_table

    return build_supported_action_sets_from_table(
        support_table,
        list(map(str, products)),
        minimum_depth=support_spec.minimum_depth,
        maximum_depth=support_spec.maximum_depth,
        minimum_observations=support_spec.minimum_observations,
        minimum_panels=support_spec.minimum_panels,
        maximum_positive_actions=support_spec.maximum_positive_actions,
    )


def coerce_action_sets(raw: Any, products: Sequence[str]) -> dict[str, tuple[float, ...]]:
    """Read action sets from the existing pickle or a plain dictionary."""

    if isinstance(raw, dict) and "supported_action_sets" in raw:
        raw = raw["supported_action_sets"]
    if not isinstance(raw, Mapping):
        raise TypeError("Action-set artifact must be a mapping or contain 'supported_action_sets'.")

    normalized = {
        normalize_identifier([key]).iloc[0]: value
        for key, value in raw.items()
    }
    output: dict[str, tuple[float, ...]] = {}
    for upc in map(str, products):
        values = normalized.get(upc)
        if values is None:
            raise KeyError(f"No supported action set found for product {upc}.")
        numeric = sorted({round(float(value), 6) for value in values if float(value) >= 0})
        if 0.0 not in numeric:
            numeric.insert(0, 0.0)
        output[upc] = tuple(numeric)
    return output


def _cost_from_margin(frame: pd.DataFrame) -> pd.Series:
    margin = pd.to_numeric(frame["gross_margin_pct_observed"], errors="coerce")
    price = pd.to_numeric(frame["model_unit_price"], errors="coerce")
    # Dominick's margin field can appear either as a fraction or percentage.
    margin_fraction = margin.where(margin.abs().le(1.5), margin / 100.0)
    return price * (1.0 - margin_fraction)


def _normalize_profile_inputs(
    selected_sample: pd.DataFrame,
    demand_predictions: pd.DataFrame,
    products: Sequence[str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = selected_sample.copy()
    predictions = demand_predictions.copy()
    sample.columns = sample.columns.astype(str).str.strip().str.lower()
    predictions.columns = predictions.columns.astype(str).str.strip().str.lower()
    for frame in (sample, predictions):
        frame["upc"] = normalize_identifier(frame["upc"])
        frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype(
            "Int64"
        )
    if "split" in predictions.columns:
        predictions = predictions.loc[
            predictions["split"].astype(str).str.lower().eq("test")
        ]
    if "model" in predictions.columns and model_name in set(
        predictions["model"].astype(str)
    ):
        predictions = predictions.loc[
            predictions["model"].astype(str).eq(model_name)
        ]
    product_set = set(map(str, products))
    return (
        sample.loc[sample["upc"].isin(product_set)].copy(),
        predictions.loc[predictions["upc"].isin(product_set)].copy(),
    )


def _select_profile_weeks(
    sample: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    total_weeks: int,
    start_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    candidate_weeks = sorted(
        int(value) for value in predictions["week"].dropna().unique().tolist()
    )
    selected_weeks = candidate_weeks[start_offset : start_offset + total_weeks]
    if len(selected_weeks) < total_weeks:
        raise ValueError(
            f"Need {total_weeks} held-out weeks from offset {start_offset}, "
            f"found {len(selected_weeks)}."
        )
    return (
        sample.loc[sample["week"].isin(selected_weeks)].copy(),
        predictions.loc[predictions["week"].isin(selected_weeks)].copy(),
        selected_weeks,
    )


def _weekly_demand_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if "mu_hat_regular_counterfactual" not in predictions.columns:
        raise ValueError(
            "Predictions lack the regular-state PPML counterfactual. "
            "Re-run demand estimation before policy optimization."
        )
    return (
        predictions.groupby(["upc", "week"], observed=True)[
            "mu_hat_regular_counterfactual"
        ]
        .sum()
        .rename("counterfactual_baseline_demand")
        .reset_index()
    )


def _weekly_price_cost_summary(sample: pd.DataFrame) -> pd.DataFrame:
    required = {
        "upc",
        "week",
        "regular_price",
        "model_unit_price",
        "gross_margin_pct_observed",
    }
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(f"Selected-sample file is missing columns: {sorted(missing)}")
    sample = sample.copy()
    sample["implied_unit_cost"] = _cost_from_margin(sample)
    return (
        sample.groupby(["upc", "week"], observed=True)
        .agg(
            regular_price_week=("regular_price", "median"),
            unit_cost_week=("implied_unit_cost", "median"),
            panel_count=("upc", "size"),
        )
        .reset_index()
    )


def _complete_weekly_profile(
    products: Sequence[str],
    selected_weeks: list[int],
    demand_weekly: pd.DataFrame,
    price_cost_weekly: pd.DataFrame,
) -> pd.DataFrame:
    full_index = pd.MultiIndex.from_product(
        [list(map(str, products)), selected_weeks],
        names=["upc", "week"],
    )
    profile = (
        pd.DataFrame(index=full_index)
        .reset_index()
        .merge(demand_weekly, on=["upc", "week"], how="left")
        .merge(price_cost_weekly, on=["upc", "week"], how="left")
    )
    profile["planning_week"] = profile["week"].map(
        {week: index + 1 for index, week in enumerate(selected_weeks)}
    )
    level_columns = [
        "counterfactual_baseline_demand",
        "regular_price_week",
        "unit_cost_week",
    ]
    for column in level_columns:
        profile[column] = pd.to_numeric(profile[column], errors="coerce")
        profile[column] = profile.groupby("upc", observed=True)[column].transform(
            lambda values: values.interpolate(limit_direction="both")
        )
    factor_map = {
        "price_factor": "regular_price_week",
        "cost_factor": "unit_cost_week",
    }
    for factor, level in factor_map.items():
        profile[factor] = profile.groupby("upc", observed=True)[level].transform(
            lambda values: values / values.median()
        )
        profile[factor] = (
            profile[factor]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .clip(lower=0.05)
        )
    profile["demand_factor"] = 1.0  # Inactive compatibility field; baseline is absolute.
    return profile


def _profile_arrays(profile: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for upc, group in profile.groupby("upc", observed=True, sort=True):
        group = group.sort_values("planning_week")
        arrays[str(upc)] = {
            "baseline_demand": group["counterfactual_baseline_demand"].to_numpy(dtype=float),
            # Compatibility only.  Evaluation uses ``baseline_demand`` directly.
            "demand_factor": group["demand_factor"].to_numpy(dtype=float),
            "price_factor": group["price_factor"].to_numpy(dtype=float),
            "cost_factor": group["cost_factor"].to_numpy(dtype=float),
            "source_week": group["week"].to_numpy(dtype=int),
        }
    return arrays


def build_weekly_economic_profiles(
    selected_sample: pd.DataFrame,
    demand_predictions: pd.DataFrame,
    products: Sequence[str],
    decision_horizon: int,
    maximum_washout: int,
    start_test_week_offset: int = 0,
    model_name: str = "product_promotion",
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], list[int]]:
    """Construct held-out weekly demand, price, and cost profile factors."""
    sample, predictions = _normalize_profile_inputs(
        selected_sample,
        demand_predictions,
        products,
        model_name,
    )
    sample, predictions, selected_weeks = _select_profile_weeks(
        sample,
        predictions,
        total_weeks=decision_horizon + maximum_washout,
        start_offset=int(start_test_week_offset),
    )
    profile = _complete_weekly_profile(
        products,
        selected_weeks,
        _weekly_demand_summary(predictions),
        _weekly_price_cost_summary(sample),
    )
    return profile, _profile_arrays(profile), selected_weeks


def constant_weekly_profiles(
    products: Sequence[str], total_weeks: int
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], list[int]]:
    """Fallback profiles that reproduce the previous constant-economics specification."""

    rows = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    source_weeks = list(range(1, total_weeks + 1))
    for upc in map(str, products):
        arrays[upc] = {
            # Absent a PPML counterfactual table, retain the draw-level fallback.
            # This is deliberately not a normalized factor.
            "demand_factor": np.ones(total_weeks),
            "price_factor": np.ones(total_weeks),
            "cost_factor": np.ones(total_weeks),
            "source_week": np.asarray(source_weeks, dtype=int),
        }
        for week in source_weeks:
            rows.append(
                {
                    "upc": upc,
                    "week": week,
                    "planning_week": week,
                    "demand_factor": 1.0,
                    "price_factor": 1.0,
                    "cost_factor": 1.0,
                }
            )
    return pd.DataFrame(rows), arrays, source_weeks


def enumerate_feasible_schedules(
    actions: Sequence[float],
    decision_horizon: int,
    cooldown: int,
    max_promotions: int,
) -> np.ndarray:
    """Enumerate all schedules; cooldown=2 means two non-promotion weeks after a promotion."""

    positive_actions = tuple(sorted({float(value) for value in actions if float(value) > 0}))
    schedules: list[np.ndarray] = [np.zeros(decision_horizon, dtype=np.float32)]
    if not positive_actions:
        return np.vstack(schedules)

    for promotion_count in range(1, max_promotions + 1):
        reduced_horizon = decision_horizon - cooldown * (promotion_count - 1)
        if reduced_horizon < promotion_count:
            break
        for reduced_weeks in combinations(range(reduced_horizon), promotion_count):
            weeks = tuple(
                reduced_week + cooldown * position
                for position, reduced_week in enumerate(reduced_weeks)
            )
            for depths in product(positive_actions, repeat=promotion_count):
                schedule = np.zeros(decision_horizon, dtype=np.float32)
                schedule[list(weeks)] = np.asarray(depths, dtype=np.float32)
                schedules.append(schedule)
    return np.vstack(schedules)


def schedule_occupancy_mask(schedule: np.ndarray) -> int:
    mask = 0
    for week, value in enumerate(np.asarray(schedule)):
        if value > 0:
            mask |= 1 << week
    return int(mask)


def occupancy_matrix_from_masks(masks: Sequence[int], horizon: int) -> np.ndarray:
    masks_array = np.asarray(masks, dtype=np.int64)
    return np.column_stack(
        [((masks_array >> week) & 1).astype(np.int8) for week in range(horizon)]
    )


def _evaluate_schedule_batch(
    schedules: np.ndarray,
    draws: Mapping[str, np.ndarray],
    profile: Mapping[str, np.ndarray],
    planning: PlanningSpec,
    *,
    add_new_promotion_displacement: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate calendars from the common fresh-start state.

    With ``add_new_promotion_displacement=False``, πN permits inherited state
    to decay but candidate promotions do not create new displacement.
    """
    n_schedules = schedules.shape[0]
    n_draws = len(draws["weights"])
    state = np.zeros((n_schedules, n_draws), dtype=np.float64)
    intercept = np.zeros(n_schedules, dtype=np.float64)
    exposure = np.zeros(n_schedules, dtype=np.float64)

    epsilon = np.asarray(draws["epsilon"], dtype=float)[None, :]
    gamma = np.asarray(draws["gamma"], dtype=float)[None, :]
    psi = np.asarray(draws["psi"], dtype=float)[None, :]
    persistence = np.asarray(draws["r"], dtype=float)[None, :]
    weights = np.asarray(draws["weights"], dtype=float)
    base_demand = np.asarray(draws["base_demand"], dtype=float)[None, :]
    regular_price = np.asarray(draws["regular_price"], dtype=float)[None, :]
    unit_cost = np.asarray(draws["unit_cost"], dtype=float)[None, :]

    for week in range(planning.evaluation_horizon):
        if week < planning.decision_horizon:
            depth = schedules[:, week][:, None].astype(float)
        else:
            depth = np.zeros((n_schedules, 1), dtype=float)
        promotion = depth > 0
        absolute_baseline = np.asarray(profile.get("baseline_demand", []), dtype=float)
        baseline = (absolute_baseline[week] if len(absolute_baseline) else base_demand)
        price_factor = float(profile["price_factor"][week])
        cost_factor = float(profile["cost_factor"][week])

        demand = (
            baseline
            * np.power(np.clip(1.0 - depth, 1e-8, None), -epsilon)
            * np.exp(gamma * promotion - psi * state)
        )
        price = regular_price * price_factor
        cost = unit_cost * cost_factor
        current_intercept = (price * (1.0 - depth) - cost) * demand
        current_exposure = regular_price * depth * demand
        discount = planning.discount_factor**week
        intercept += discount * (current_intercept @ weights)
        exposure += discount * (current_exposure @ weights)
        state = persistence * state + (depth if add_new_promotion_displacement else 0.0)

    terminal_state = state @ weights
    return intercept, exposure, terminal_state


def evaluate_schedules(
    schedules: np.ndarray,
    draws: Mapping[str, np.ndarray],
    profile: Mapping[str, np.ndarray],
    planning: PlanningSpec,
    batch_size: int = 256,
    *,
    add_new_promotion_displacement: bool = True,
) -> pd.DataFrame:
    """Evaluate schedule values as affine functions of reimbursement share."""

    rows: list[pd.DataFrame] = []
    for start in range(0, len(schedules), batch_size):
        stop = min(start + batch_size, len(schedules))
        intercept, exposure, terminal_state = _evaluate_schedule_batch(
            schedules[start:stop], draws, profile, planning,
            add_new_promotion_displacement=add_new_promotion_displacement,
        )
        rows.append(
            pd.DataFrame(
                {
                    "schedule_index": np.arange(start, stop, dtype=int),
                    "intercept": intercept,
                    "exposure": exposure,
                    "terminal_state": terminal_state,
                }
            )
        )
    values = pd.concat(rows, ignore_index=True)
    values["promotion_count"] = (schedules > 0).sum(axis=1).astype(int)
    values["occupancy_mask"] = [schedule_occupancy_mask(row) for row in schedules]
    return values


def audit_draw_weighted_schedule_values(
    schedule_system: Mapping[str, Any],
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    *,
    upc: str,
    alpha: float,
    capacity: int,
) -> pd.DataFrame:
    """Return per-draw candidate values and verify the stored weighted objective."""
    product = str(upc)
    draws = draws_by_product[product]
    weights = np.asarray(draws["weights"], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("Behavioral draw weights must sum to one.")
    schedules = schedule_system["product_artifacts"][product]["schedules"]
    stored = schedule_system["product_artifacts"][product]["schedule_values"].copy()
    selected = solve_dynamic_category(schedule_system, alpha=float(alpha), capacity=int(capacity))
    selected_indices = set(
        selected["selected_candidates"].loc[
            selected["selected_candidates"]["upc"].eq(product), "schedule_index"
        ].astype(int)
    )
    rows: list[pd.DataFrame] = []
    add_new = bool(schedule_system.get("add_new_promotion_displacement", True))
    for draw_id, weight in enumerate(weights):
        one_draw = {key: np.asarray(value, dtype=float).copy() for key, value in draws.items()}
        one_draw["weights"] = np.zeros_like(weights)
        one_draw["weights"][draw_id] = 1.0
        values = evaluate_schedules(
            schedules, one_draw, weekly_profiles[product], schedule_system["planning"],
            add_new_promotion_displacement=add_new,
        )
        rows.append(pd.DataFrame({
            "draw_id": draw_id,
            "draw_weight": weight,
            "schedule_index": values["schedule_index"],
            "per_draw_value": values["intercept"] + float(alpha) * values["exposure"],
        }))
    audit = pd.concat(rows, ignore_index=True)
    weighted = audit.assign(weighted_value=lambda x: x.draw_weight * x.per_draw_value).groupby(
        "schedule_index", observed=True
    )["weighted_value"].sum()
    stored_value = stored.set_index("schedule_index")["intercept"] + float(alpha) * stored.set_index("schedule_index")["exposure"]
    np.testing.assert_allclose(weighted.loc[stored_value.index], stored_value, atol=1e-10, rtol=1e-10)
    audit["weighted_expected_schedule_value"] = audit["schedule_index"].map(weighted)
    audit["stored_schedule_value"] = audit["schedule_index"].map(stored_value)
    audit["selected_by_optimizer"] = audit["schedule_index"].isin(selected_indices)
    return audit.sort_values(["schedule_index", "draw_id"]).reset_index(drop=True)


def prune_product_candidates(
    schedule_values: pd.DataFrame,
    alpha_grid: Sequence[float],
) -> pd.DataFrame:
    """Keep the best schedule for each occupancy pattern at every evaluated alpha."""

    alphas = np.asarray(sorted({float(value) for value in alpha_grid}), dtype=float)
    retained: list[int] = []
    for _, group in schedule_values.groupby("occupancy_mask", observed=True, sort=False):
        intercept = group["intercept"].to_numpy(dtype=float)
        exposure = group["exposure"].to_numpy(dtype=float)
        values = intercept[:, None] + exposure[:, None] * alphas[None, :]
        local_positions = np.unique(np.argmax(values, axis=0))
        retained.extend(group.iloc[local_positions].index.tolist())
    output = schedule_values.loc[sorted(set(retained))].copy().reset_index(drop=True)
    output["candidate_index"] = np.arange(len(output), dtype=int)
    return output



def schedule_input_fingerprint(
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    action_sets: Mapping[str, Sequence[float]],
    planning: PlanningSpec,
    alpha_grid: Sequence[float],
    *,
    add_new_promotion_displacement: bool = True,
) -> str:
    """Hash all inputs that determine schedule values and occupancy pruning."""

    digest = hashlib.sha1()
    digest.update(json.dumps(asdict(planning), sort_keys=True).encode("utf-8"))
    digest.update(np.asarray(sorted({float(value) for value in alpha_grid}), dtype=float).tobytes())
    digest.update(str(bool(add_new_promotion_displacement)).encode("ascii"))

    for upc in sorted(draws_by_product):
        digest.update(str(upc).encode("utf-8"))
        digest.update(
            np.asarray(
                [float(value) for value in action_sets[upc]],
                dtype=float,
            ).tobytes()
        )
        for key in (
            "epsilon",
            "gamma",
            "psi",
            "r",
            "weights",
            "base_demand",
            "regular_price",
            "unit_cost",
        ):
            digest.update(np.asarray(draws_by_product[upc][key], dtype=float).tobytes())
        for key in ("baseline_demand", "price_factor", "cost_factor"):
            if key not in weekly_profiles[upc]:
                # Constant fallback profiles intentionally use draw-level demand.
                if key == "baseline_demand":
                    continue
            values = np.asarray(weekly_profiles[upc][key], dtype=float)
            digest.update(values[: planning.evaluation_horizon].tobytes())

    return digest.hexdigest()

def build_schedule_system(
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    action_sets: Mapping[str, Sequence[float]],
    planning: PlanningSpec,
    alpha_grid: Sequence[float],
    batch_size: int = 256,
    *,
    add_new_promotion_displacement: bool = True,
) -> dict[str, Any]:
    """Enumerate, evaluate, and occupancy-prune product schedules."""

    input_fingerprint = schedule_input_fingerprint(
        draws_by_product=draws_by_product,
        weekly_profiles=weekly_profiles,
        action_sets=action_sets,
        planning=planning,
        alpha_grid=alpha_grid,
        add_new_promotion_displacement=add_new_promotion_displacement,
    )

    product_artifacts: dict[str, dict[str, Any]] = {}
    for upc in sorted(draws_by_product):
        if upc not in action_sets:
            raise KeyError(f"Missing action set for product {upc}.")
        if upc not in weekly_profiles:
            raise KeyError(f"Missing weekly economic profile for product {upc}.")
        schedules = enumerate_feasible_schedules(
            action_sets[upc],
            decision_horizon=planning.decision_horizon,
            cooldown=planning.cooldown,
            max_promotions=planning.max_promotions,
        )
        values = evaluate_schedules(
            schedules,
            draws_by_product[upc],
            weekly_profiles[upc],
            planning,
            batch_size=batch_size,
            add_new_promotion_displacement=add_new_promotion_displacement,
        )
        candidates = prune_product_candidates(values, alpha_grid)
        product_artifacts[upc] = {
            "actions": tuple(float(value) for value in action_sets[upc]),
            "schedules": schedules,
            "schedule_values": values,
            "candidates": candidates,
        }

    return {
        "planning": planning,
        "add_new_promotion_displacement": bool(add_new_promotion_displacement),
        "action_sets": {key: tuple(value) for key, value in action_sets.items()},
        "products": sorted(product_artifacts),
        "product_artifacts": product_artifacts,
        "alpha_grid": np.asarray(alpha_grid, dtype=float),
        "input_fingerprint": input_fingerprint,
    }


def _category_candidate_table(
    schedule_system: Mapping[str, Any], alpha: float
) -> tuple[pd.DataFrame, np.ndarray]:
    horizon = schedule_system["planning"].decision_horizon
    frames: list[pd.DataFrame] = []
    occupancy_blocks: list[np.ndarray] = []
    offset = 0
    for product_position, upc in enumerate(schedule_system["products"]):
        candidates = schedule_system["product_artifacts"][upc]["candidates"].copy()
        candidates["upc"] = upc
        candidates["product_position"] = product_position
        candidates["global_index"] = np.arange(offset, offset + len(candidates), dtype=int)
        candidates["value"] = candidates["intercept"] + alpha * candidates["exposure"]
        occupancy = occupancy_matrix_from_masks(candidates["occupancy_mask"], horizon)
        frames.append(candidates)
        occupancy_blocks.append(occupancy)
        offset += len(candidates)
    return pd.concat(frames, ignore_index=True), np.vstack(occupancy_blocks)


def solve_dynamic_category(
    schedule_system: Mapping[str, Any],
    alpha: float,
    capacity: int,
    compute_second_best: bool = True,
    time_limit_seconds: float | None = None,
) -> dict[str, Any]:
    """Solve the multiple-choice category schedule MILP."""

    candidates, occupancy = _category_candidate_table(schedule_system, alpha)
    products = list(schedule_system["products"])
    n_variables = len(candidates)
    product_rows = np.zeros((len(products), n_variables), dtype=float)
    for position in range(len(products)):
        product_rows[position, candidates["product_position"].eq(position).to_numpy()] = 1.0
    capacity_rows = occupancy.T.astype(float)

    matrix = csc_matrix(np.vstack([product_rows, capacity_rows]))
    lower = np.concatenate([np.ones(len(products)), np.full(capacity_rows.shape[0], -np.inf)])
    upper = np.concatenate([np.ones(len(products)), np.full(capacity_rows.shape[0], capacity)])
    constraints = LinearConstraint(matrix, lower, upper)
    options: dict[str, Any] = {"disp": False}
    if time_limit_seconds is not None:
        options["time_limit"] = float(time_limit_seconds)

    result = milp(
        c=-candidates["value"].to_numpy(dtype=float),
        integrality=np.ones(n_variables, dtype=int),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=constraints,
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Dynamic category MILP failed: {result.message}")

    selected_positions = np.flatnonzero(result.x > 0.5)
    selected = candidates.iloc[selected_positions].copy().sort_values("product_position")
    best_value = float(selected["value"].sum())
    second_best_value = np.nan
    second_selected: pd.DataFrame | None = None

    if compute_second_best:
        exclusion_row = np.zeros(n_variables, dtype=float)
        exclusion_row[selected_positions] = 1.0
        matrix_second = vstack([matrix, csc_matrix(exclusion_row.reshape(1, -1))], format="csc")
        lower_second = np.concatenate([lower, [-np.inf]])
        upper_second = np.concatenate([upper, [len(products) - 1]])
        result_second = milp(
            c=-candidates["value"].to_numpy(dtype=float),
            integrality=np.ones(n_variables, dtype=int),
            bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
            constraints=LinearConstraint(matrix_second, lower_second, upper_second),
            options=options,
        )
        if result_second.success and result_second.x is not None:
            second_positions = np.flatnonzero(result_second.x > 0.5)
            second_selected = (
                candidates.iloc[second_positions]
                .copy()
                .sort_values("product_position")
            )
            second_best_value = float(second_selected["value"].sum())

    def to_schedule_map(rows: pd.DataFrame) -> dict[str, np.ndarray]:
        schedule_map: dict[str, np.ndarray] = {}
        for row in rows.itertuples(index=False):
            schedule_map[str(row.upc)] = schedule_system["product_artifacts"][str(row.upc)][
                "schedules"
            ][int(row.schedule_index)].astype(float)
        return schedule_map

    schedule_map = to_schedule_map(selected)
    second_schedule_map = (
        to_schedule_map(second_selected) if second_selected is not None else None
    )

    return {
        "selected_candidates": selected.reset_index(drop=True),
        "schedule_map": schedule_map,
        "second_selected_candidates": (
            second_selected.reset_index(drop=True)
            if second_selected is not None
            else None
        ),
        "second_schedule_map": second_schedule_map,
        "best_value": best_value,
        "second_best_value": second_best_value,
        "best_second_gap": best_value - second_best_value
        if np.isfinite(second_best_value)
        else np.nan,
        "solver_message": str(result.message),
    }


def _expected_current_profit(
    depth: float,
    state: np.ndarray,
    draws: Mapping[str, np.ndarray],
    profile: Mapping[str, np.ndarray],
    week: int,
    alpha: float,
) -> float:
    depth_value = float(depth)
    promotion = depth_value > 0
    baseline = np.asarray(profile.get("baseline_demand", draws["base_demand"]), dtype=float)
    baseline = baseline[week] if baseline.ndim else baseline
    demand = (
        baseline
        * np.power(max(1.0 - depth_value, 1e-8), -draws["epsilon"])
        * np.exp(draws["gamma"] * promotion - draws["psi"] * state)
    )
    price = draws["regular_price"] * float(profile["price_factor"][week])
    cost = draws["unit_cost"] * float(profile["cost_factor"][week])
    reimbursement = alpha * draws["regular_price"] * depth_value
    profit = (price * (1.0 - depth_value) - cost + reimbursement) * demand
    return float(profit @ draws["weights"])


def simulate_myopic_category(
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    action_sets: Mapping[str, Sequence[float]],
    planning: PlanningSpec,
    alpha: float,
    capacity: int,
) -> dict[str, Any]:
    """Simulate a capacity-constrained myopic policy with draw-aligned states."""

    products = sorted(draws_by_product)
    states = {
        upc: np.zeros(len(draws_by_product[upc]["weights"]), dtype=float) for upc in products
    }
    last_promotion = {upc: -10_000 for upc in products}
    promotion_count = {upc: 0 for upc in products}
    schedules = {upc: np.zeros(planning.decision_horizon, dtype=float) for upc in products}

    weekly_profit_rows: list[dict[str, Any]] = []
    product_profit = {upc: 0.0 for upc in products}

    for week in range(planning.evaluation_horizon):
        chosen = {upc: 0.0 for upc in products}
        if week < planning.decision_horizon:
            candidates: list[tuple[float, str, float]] = []
            for upc in products:
                draws = draws_by_product[upc]
                profile = weekly_profiles[upc]
                base_profit = _expected_current_profit(
                    0.0, states[upc], draws, profile, week, alpha
                )
                feasible_positive = (
                    promotion_count[upc] < planning.max_promotions
                    and week - last_promotion[upc] > planning.cooldown
                )
                if feasible_positive:
                    positive_actions = [
                        float(value) for value in action_sets[upc] if float(value) > 0
                    ]
                    if positive_actions:
                        action_profits = [
                            _expected_current_profit(
                                action, states[upc], draws, profile, week, alpha
                            )
                            for action in positive_actions
                        ]
                        best_position = int(np.argmax(action_profits))
                        best_action = positive_actions[best_position]
                        incremental = action_profits[best_position] - base_profit
                        if incremental > 1e-12:
                            candidates.append((float(incremental), upc, best_action))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _, upc, action in candidates[: int(capacity)]:
                chosen[upc] = action
                schedules[upc][week] = action
                last_promotion[upc] = week
                promotion_count[upc] += 1

        for upc in products:
            draws = draws_by_product[upc]
            profile = weekly_profiles[upc]
            depth = chosen[upc]
            current_profit = _expected_current_profit(
                depth, states[upc], draws, profile, week, alpha
            )
            discounted_profit = (planning.discount_factor**week) * current_profit
            product_profit[upc] += discounted_profit
            weekly_profit_rows.append(
                {
                    "upc": upc,
                    "week": week + 1,
                    "decision_week": week < planning.decision_horizon,
                    "action": depth,
                    "profit": discounted_profit,
                }
            )
            states[upc] = draws["r"] * states[upc] + depth

    terminal_state = {
        upc: float(states[upc] @ draws_by_product[upc]["weights"]) for upc in products
    }
    return {
        "schedule_map": schedules,
        "weekly_profit": pd.DataFrame(weekly_profit_rows),
        "product_profit": pd.Series(product_profit, name="myopic_profit"),
        "total_profit": float(sum(product_profit.values())),
        "terminal_state": terminal_state,
    }


def evaluate_schedule_map(
    schedule_map: Mapping[str, Sequence[float]],
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    planning: PlanningSpec,
    alpha: float,
) -> dict[str, Any]:
    """Evaluate a fixed category schedule and return exact additive details."""

    rows: list[dict[str, Any]] = []
    product_profit: dict[str, float] = {}
    terminal_state: dict[str, float] = {}

    for upc in sorted(draws_by_product):
        draws = draws_by_product[upc]
        profile = weekly_profiles[upc]
        schedule = np.asarray(schedule_map[upc], dtype=float)
        state = np.zeros(len(draws["weights"]), dtype=float)
        total = 0.0
        for week in range(planning.evaluation_horizon):
            depth = float(schedule[week]) if week < planning.decision_horizon else 0.0
            promotion = depth > 0
            baseline = np.asarray(profile.get("baseline_demand", draws["base_demand"]), dtype=float)
            baseline = baseline[week] if baseline.ndim else baseline
            demand = (
                baseline
                * np.power(max(1.0 - depth, 1e-8), -draws["epsilon"])
                * np.exp(draws["gamma"] * promotion - draws["psi"] * state)
            )
            price = draws["regular_price"] * float(profile["price_factor"][week])
            cost = draws["unit_cost"] * float(profile["cost_factor"][week])
            reimbursement = alpha * draws["regular_price"] * depth
            profit_draw = (price * (1.0 - depth) - cost + reimbursement) * demand
            expected_profit = float(profit_draw @ draws["weights"])
            discounted_profit = (planning.discount_factor**week) * expected_profit
            total += discounted_profit
            rows.append(
                {
                    "upc": upc,
                    "week": week + 1,
                    "decision_week": week < planning.decision_horizon,
                    "action": depth,
                    "profit": discounted_profit,
                    "expected_demand": float(demand @ draws["weights"]),
                    "inventory_state_before": float(state @ draws["weights"]),
                }
            )
            state = draws["r"] * state + depth
        product_profit[upc] = total
        terminal_state[upc] = float(state @ draws["weights"])

    return {
        "weekly_profit": pd.DataFrame(rows),
        "product_profit": pd.Series(product_profit, name="dynamic_profit"),
        "total_profit": float(sum(product_profit.values())),
        "terminal_state": terminal_state,
    }


def demand_multiplier_audit(
    schedule_map: Mapping[str, Sequence[float]],
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    planning: PlanningSpec,
) -> pd.DataFrame:
    """Expose the multiplier chain used to replay a policy calendar."""
    rows: list[dict[str, object]] = []
    for upc, draws in draws_by_product.items():
        weights = np.asarray(draws["weights"], dtype=float)
        state = np.zeros_like(weights)
        schedule = np.asarray(schedule_map[upc], dtype=float)
        for week in range(planning.evaluation_horizon):
            depth = float(schedule[week]) if week < planning.decision_horizon else 0.0
            profile = weekly_profiles[upc]
            baseline = np.asarray(profile.get("baseline_demand", draws["base_demand"]), dtype=float)
            q_base = baseline[week] if baseline.ndim else baseline
            # PPML product-week baselines are scalar levels; broadcast them so
            # the draw-weighted multiplier diagnostic remains well defined.
            if np.ndim(q_base) == 0:
                q_base = np.full_like(weights, float(q_base), dtype=float)
            price = np.power(1.0 - depth, -np.asarray(draws["epsilon"], dtype=float))
            lift = np.exp(np.asarray(draws["gamma"], dtype=float) * (depth > 0))
            displacement = np.exp(-np.asarray(draws["psi"], dtype=float) * state)
            expected_demand = float((q_base * price * lift * displacement) @ weights)
            rows.append({"upc": str(upc), "week": week + 1, "depth": depth,
                         "q_base": float(q_base @ weights), "price_response": float(price @ weights),
                         "promotion_lift": float(lift @ weights),
                         "displacement_multiplier": float(displacement @ weights),
                         "multiplier_product_demand": expected_demand,
                         "predicted_demand": expected_demand,
                         "prediction_reconstruction_error": 0.0})
            state = np.asarray(draws["r"], dtype=float) * state + depth
    return pd.DataFrame(rows)


def schedule_signature(schedule_map: Mapping[str, Sequence[float]], decimals: int = 4) -> str:
    payload = {
        str(upc): [round(float(value), decimals) for value in schedule]
        for upc, schedule in sorted(schedule_map.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def active_products(schedule_map: Mapping[str, Sequence[float]]) -> tuple[str, ...]:
    return tuple(
        upc
        for upc, schedule in sorted(schedule_map.items())
        if np.any(np.asarray(schedule, dtype=float) > 0)
    )


def policy_disagreement_summary(
    dynamic_schedule: Mapping[str, Sequence[float]],
    myopic_schedule: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    products = sorted(dynamic_schedule)
    dynamic_matrix = np.vstack([np.asarray(dynamic_schedule[upc], dtype=float) for upc in products])
    myopic_matrix = np.vstack([np.asarray(myopic_schedule[upc], dtype=float) for upc in products])
    dynamic_status = dynamic_matrix > 0
    myopic_status = myopic_matrix > 0
    status_difference = dynamic_status != myopic_status
    depth_difference = dynamic_status & myopic_status & ~np.isclose(dynamic_matrix, myopic_matrix)
    action_difference = ~np.isclose(dynamic_matrix, myopic_matrix)

    timing_shift_products = 0
    same_action_multiset_products = 0
    for position, upc in enumerate(products):
        dynamic_positive = sorted(dynamic_matrix[position][dynamic_matrix[position] > 0].round(6))
        myopic_positive = sorted(myopic_matrix[position][myopic_matrix[position] > 0].round(6))
        if dynamic_positive == myopic_positive:
            same_action_multiset_products += 1
            if not np.allclose(dynamic_matrix[position], myopic_matrix[position]):
                timing_shift_products += 1

    return {
        "action_disagreements": int(action_difference.sum()),
        "status_disagreements": int(status_difference.sum()),
        "depth_disagreements": int(depth_difference.sum()),
        "disagreement_rate": float(action_difference.mean()),
        "timing_shift_products": int(timing_shift_products),
        "same_action_multiset_products": int(same_action_multiset_products),
        "dynamic_active_products": len(active_products(dynamic_schedule)),
        "myopic_active_products": len(active_products(myopic_schedule)),
        "active_set_equal": active_products(dynamic_schedule) == active_products(myopic_schedule),
    }


def capacity_usage(schedule_map: Mapping[str, Sequence[float]]) -> np.ndarray:
    return np.sum(
        np.vstack([np.asarray(schedule, dtype=float) > 0 for schedule in schedule_map.values()]),
        axis=0,
    ).astype(int)


def chosen_action_support(
    schedule_map: Mapping[str, Sequence[float]], support_table: pd.DataFrame
) -> pd.DataFrame:
    support = prepare_support_table(support_table)
    rows: list[dict[str, Any]] = []
    for upc, schedule in sorted(schedule_map.items()):
        for week, depth in enumerate(np.asarray(schedule, dtype=float), start=1):
            if depth <= 0:
                continue
            product_support = support.loc[support["upc"].eq(str(upc))].copy()
            if product_support.empty:
                rows.append(
                    {
                        "upc": str(upc),
                        "week": week,
                        "depth": depth,
                        "matched_depth": np.nan,
                        "observations": np.nan,
                        "panels": np.nan,
                    }
                )
                continue
            position = (product_support["depth_cluster"] - depth).abs().idxmin()
            match = product_support.loc[position]
            rows.append(
                {
                    "upc": str(upc),
                    "week": week,
                    "depth": depth,
                    "matched_depth": float(match["depth_cluster"]),
                    "observations": int(match["observations"]),
                    "panels": int(match["panels"]),
                }
            )
    return pd.DataFrame(rows)


def analyze_policy_pair(
    dynamic_solution: Mapping[str, Any],
    myopic_solution: Mapping[str, Any],
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    planning: PlanningSpec,
    support_table: pd.DataFrame,
    alpha: float,
    capacity: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    dynamic_detail = evaluate_schedule_map(
        dynamic_solution["schedule_map"],
        draws_by_product,
        weekly_profiles,
        planning,
        alpha,
    )
    myopic_detail = myopic_solution
    dynamic_total = float(dynamic_detail["total_profit"])
    myopic_total = float(myopic_detail["total_profit"])
    vdo = dynamic_total - myopic_total

    disagreement = policy_disagreement_summary(
        dynamic_solution["schedule_map"], myopic_solution["schedule_map"]
    )
    dynamic_usage = capacity_usage(dynamic_solution["schedule_map"])
    myopic_usage = capacity_usage(myopic_solution["schedule_map"])
    support = chosen_action_support(dynamic_solution["schedule_map"], support_table)

    summary = {
        "reimbursement_share": float(alpha),
        "alpha": float(alpha),  # Compatibility alias for pre-rerun artifacts.
        "capacity": int(capacity),
        "decision_horizon": planning.decision_horizon,
        "washout_horizon": planning.washout_horizon,
        "evaluation_horizon": planning.evaluation_horizon,
        "dynamic_profit": dynamic_total,
        "myopic_profit": myopic_total,
        "vdo": vdo,
        "vdo_percent": 100.0 * vdo / myopic_total if myopic_total != 0 else np.nan,
        "dynamic_promotion_count": int(dynamic_usage.sum()),
        "myopic_promotion_count": int(myopic_usage.sum()),
        "dynamic_binding_weeks": int(np.sum(dynamic_usage >= capacity)),
        "myopic_binding_weeks": int(np.sum(myopic_usage >= capacity)),
        "dynamic_active_set": "|".join(active_products(dynamic_solution["schedule_map"])),
        "myopic_active_set": "|".join(active_products(myopic_solution["schedule_map"])),
        "dynamic_schedule_signature": schedule_signature(dynamic_solution["schedule_map"]),
        "myopic_schedule_signature": schedule_signature(myopic_solution["schedule_map"]),
        "best_second_gap": float(dynamic_solution.get("best_second_gap", np.nan)),
        "minimum_chosen_support": float(support["observations"].min())
        if not support.empty
        else np.nan,
        "median_chosen_support": float(support["observations"].median())
        if not support.empty
        else np.nan,
        "minimum_chosen_panels": float(support["panels"].min())
        if not support.empty
        else np.nan,
        "maximum_terminal_state_dynamic": max(dynamic_detail["terminal_state"].values()),
        "maximum_terminal_state_myopic": max(myopic_detail["terminal_state"].values()),
        **disagreement,
    }

    dynamic_product = dynamic_detail["product_profit"].rename("dynamic_profit")
    myopic_product = myopic_detail["product_profit"].rename("myopic_profit")
    product_decomposition = pd.concat([dynamic_product, myopic_product], axis=1).reset_index()
    product_decomposition = product_decomposition.rename(columns={"index": "upc"})
    product_decomposition["vdo_contribution"] = (
        product_decomposition["dynamic_profit"] - product_decomposition["myopic_profit"]
    )
    for key, value in {
        "reimbursement_share": alpha,
        "alpha": alpha,
        "capacity": capacity,
        "washout_horizon": planning.washout_horizon,
    }.items():
        product_decomposition[key] = value

    dynamic_weekly = (
        dynamic_detail["weekly_profit"].groupby("week", observed=True)["profit"].sum()
    ).rename("dynamic_profit")
    myopic_weekly = (
        myopic_detail["weekly_profit"].groupby("week", observed=True)["profit"].sum()
    ).rename("myopic_profit")
    weekly_decomposition = pd.concat([dynamic_weekly, myopic_weekly], axis=1).reset_index()
    weekly_decomposition["vdo_contribution"] = (
        weekly_decomposition["dynamic_profit"] - weekly_decomposition["myopic_profit"]
    )
    weekly_decomposition["cumulative_vdo"] = weekly_decomposition[
        "vdo_contribution"
    ].cumsum()
    weekly_decomposition["decision_week"] = weekly_decomposition["week"].le(
        planning.decision_horizon
    )
    for key, value in {
        "reimbursement_share": alpha,
        "alpha": alpha,
        "capacity": capacity,
        "washout_horizon": planning.washout_horizon,
    }.items():
        weekly_decomposition[key] = value

    return summary, product_decomposition, weekly_decomposition


def run_policy_grid(
    schedule_system: Mapping[str, Any],
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    action_sets: Mapping[str, Sequence[float]],
    support_table: pd.DataFrame,
    alpha_values: Sequence[float],
    capacities: Sequence[int],
    compute_second_best: bool = True,
    time_limit_seconds: float | None = None,
) -> dict[str, Any]:
    """Run dynamic and myopic policies over a reimbursement-capacity grid."""

    planning: PlanningSpec = schedule_system["planning"]
    summaries: list[dict[str, Any]] = []
    product_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []
    schedules: dict[tuple[float, int], dict[str, Any]] = {}

    for capacity in capacities:
        for alpha in alpha_values:
            dynamic = solve_dynamic_category(
                schedule_system,
                alpha=float(alpha),
                capacity=int(capacity),
                compute_second_best=compute_second_best,
                time_limit_seconds=time_limit_seconds,
            )
            myopic = simulate_myopic_category(
                draws_by_product,
                weekly_profiles,
                action_sets,
                planning,
                alpha=float(alpha),
                capacity=int(capacity),
            )
            summary, product_decomposition, weekly_decomposition = analyze_policy_pair(
                dynamic,
                myopic,
                draws_by_product,
                weekly_profiles,
                planning,
                support_table,
                alpha=float(alpha),
                capacity=int(capacity),
            )
            summaries.append(summary)
            product_frames.append(product_decomposition)
            weekly_frames.append(weekly_decomposition)
            schedules[(round(float(alpha), 8), int(capacity))] = {
                "dynamic": dynamic["schedule_map"],
                "myopic": myopic["schedule_map"],
            }

    return {
        "results": pd.DataFrame(summaries).sort_values(["capacity", "alpha"]),
        "product_decomposition": pd.concat(product_frames, ignore_index=True),
        "weekly_decomposition": pd.concat(weekly_frames, ignore_index=True),
        "schedules": schedules,
    }


def displacement_naive_draws(
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Legacy ψ=0 shortcut retained for compatibility tests only."""
    output: dict[str, dict[str, np.ndarray]] = {}
    for upc, draws in draws_by_product.items():
        output[str(upc)] = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in draws.items()
        }
        output[str(upc)]["psi"] = np.zeros_like(output[str(upc)]["psi"])
    return output


def run_three_policy_grid(
    schedule_system: Mapping[str, Any],
    naive_schedule_system: Mapping[str, Any],
    draws_by_product: Mapping[str, Mapping[str, np.ndarray]],
    weekly_profiles: Mapping[str, Mapping[str, np.ndarray]],
    action_sets: Mapping[str, Sequence[float]],
    alpha_values: Sequence[float],
    capacities: Sequence[int],
    time_limit_seconds: float | None = None,
    compute_second_best: bool = True,
) -> dict[str, Any]:
    """Compare myopic, displacement-naive, and displacement-aware calendars.

    πN is optimized with the full ψ but without candidate-promotion state
    additions; all three calendars are replayed with the full state equation.
    """
    planning: PlanningSpec = schedule_system["planning"]
    rows: list[dict[str, Any]] = []
    schedules: dict[tuple[float, int], dict[str, Any]] = {}
    for capacity in capacities:
        for alpha in alpha_values:
            share = float(alpha)
            dynamic = solve_dynamic_category(
                schedule_system, share, int(capacity), compute_second_best=compute_second_best,
                time_limit_seconds=time_limit_seconds
            )
            naive = solve_dynamic_category(
                naive_schedule_system, share, int(capacity), compute_second_best=compute_second_best,
                time_limit_seconds=time_limit_seconds
            )
            myopic = simulate_myopic_category(
                draws_by_product, weekly_profiles, action_sets, planning, share, int(capacity)
            )
            dynamic_value = float(evaluate_schedule_map(
                dynamic["schedule_map"], draws_by_product, weekly_profiles, planning, share
            )["total_profit"])
            naive_value = float(evaluate_schedule_map(
                naive["schedule_map"], draws_by_product, weekly_profiles, planning, share
            )["total_profit"])
            myopic_value = float(myopic["total_profit"])
            if dynamic_value < naive_value - 1e-6 or dynamic_value < myopic_value - 1e-6:
                raise AssertionError("Displacement-aware optimum fails a full-objective dominance check.")
            delta_plan = naive_value - myopic_value
            delta_disp = dynamic_value - naive_value
            delta_total = dynamic_value - myopic_value
            if not np.isclose(delta_plan + delta_disp, delta_total, atol=1e-8):
                raise AssertionError("Three-policy value components do not add up.")
            rows.append({
                "reimbursement_share": share,
                "alpha": share,
                "capacity": int(capacity),
                # Paper-facing three-policy names.  Retain the legacy columns
                # below so previously generated downstream artifacts still load.
                "value_piM": myopic_value,
                "value_piN": naive_value,
                "value_piD": dynamic_value,
                "delta_plan": delta_plan,
                "delta_disp": delta_disp,
                "delta_total": delta_total,
                "dynamic_profit": dynamic_value,
                "naive_dynamic_profit": naive_value,
                "myopic_profit": myopic_value,
                "vdo": delta_total,
                "forward_planning_naive_increment": delta_plan,
                "displacement_aware_increment": delta_disp,
                "sequential_decomposition_error": (
                    delta_total - delta_plan - delta_disp
                ),
                "dynamic_schedule_signature": schedule_signature(dynamic["schedule_map"]),
                "naive_dynamic_schedule_signature": schedule_signature(naive["schedule_map"]),
                "myopic_schedule_signature": schedule_signature(myopic["schedule_map"]),
            })
            schedules[(round(share, 8), int(capacity))] = {
                "dynamic": dynamic["schedule_map"],
                "naive_dynamic": naive["schedule_map"],
                "myopic": myopic["schedule_map"],
            }
    results = pd.DataFrame(rows).sort_values(["capacity", "reimbursement_share"])
    if not np.allclose(results["sequential_decomposition_error"], 0.0, atol=1e-8):
        raise AssertionError("Three-policy decomposition does not add up.")
    return {"results": results.reset_index(drop=True), "schedules": schedules}


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def config_to_dict(planning: PlanningSpec, support: SupportSpec) -> dict[str, Any]:
    return {"planning": asdict(planning), "support": asdict(support)}
