"""Alternative-PPML robustness: product_promotion, isolated from main artifacts."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_promotion_planning.demand import POLICY_COMMON_CONTROLS, common_origin_ex_ante_predictions
from dynamic_promotion_planning.policy import (
    build_schedule_system, build_weekly_economic_profiles, load_pickle,
    run_three_policy_grid, save_pickle,
)
from dynamic_promotion_planning.policy_workflow import demand_only_profiles


MODEL = "product_promotion"
OUT = ROOT / "results" / "robustness" / "alternative_ppml_product_promotion"
ARTIFACTS = ROOT / "artifacts" / "robustness" / "alternative_ppml_product_promotion"


def _transitions(schedules: dict, capacities: list[int]) -> pd.DataFrame:
    from dynamic_promotion_planning.policy import schedule_signature
    rows = []
    aliases = {"myopic": "piM", "naive_dynamic": "piN", "dynamic": "piD"}
    for capacity in capacities:
        entries = sorted((share, item) for (share, cap), item in schedules.items() if cap == capacity)
        for key, label in aliases.items():
            signatures = [schedule_signature(item[key]) for _, item in entries]
            locations = [share for (share, _), before, after in zip(entries[1:], signatures, signatures[1:]) if before != after]
            rows.append({"capacity": capacity, "policy": label, "transition_count": len(locations),
                         "transition_locations": "|".join(f"{value:.2f}" for value in locations)})
    return pd.DataFrame(rows)


def summarize(name: str, results: pd.DataFrame, schedules: dict, capacities: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    transitions = _transitions(schedules, capacities)
    rows = []
    for capacity, group in results.groupby("capacity", observed=True):
        peak = group.loc[group["delta_total"].idxmax()]
        counts = transitions.loc[transitions["capacity"].eq(capacity)].set_index("policy")
        rows.append({"specification": name, "capacity": int(capacity), "peak_delta_total": float(peak.delta_total),
                     "peak_lambda": float(peak.reimbursement_share), "peak_delta_plan": float(peak.delta_plan),
                     "peak_delta_disp": float(peak.delta_disp), "displacement_share_at_peak": float(peak.delta_disp / peak.delta_total),
                     "piM_transitions": int(counts.loc["piM", "transition_count"]), "piN_transitions": int(counts.loc["piN", "transition_count"]), "piD_transitions": int(counts.loc["piD", "transition_count"]),
                     "piM_transition_locations": counts.loc["piM", "transition_locations"], "piN_transition_locations": counts.loc["piN", "transition_locations"], "piD_transition_locations": counts.loc["piD", "transition_locations"]})
    full = results.copy()
    full.insert(0, "specification", name)
    return pd.DataFrame(rows), full


def _source_frame(artifact: dict) -> pd.DataFrame:
    """Rebuild just the demand-model frame; all non-PPML policy inputs stay frozen."""
    selected = pd.read_parquet(ROOT / "data" / "processed" / "paper_selected_sample.parquet")
    panels = set(selected["store_upc"].astype(str))
    products = set(map(str, artifact["products"]))
    source = pd.read_parquet(ROOT / "data" / "processed" / "cereal_demand_model_data.parquet")
    source["store_upc"] = source["store_upc"].astype(str)
    source["upc"] = source["upc"].astype(str)
    frame = source.loc[source["store_upc"].isin(panels) & source["upc"].isin(products)].copy()
    state = frame["pricing_state"].astype("string").str.strip().str.lower()
    frame["promotion_indicator"] = state.eq("promotion").astype("int8")
    frame["post_promotion_indicator"] = state.eq("post_promotion").astype("int8")
    frame["log_price_model"] = np.log(pd.to_numeric(frame["model_unit_price"], errors="coerce"))
    frame["discount_depth_model"] = pd.to_numeric(frame["discount_depth"], errors="coerce").fillna(0.0)
    frame["discount_depth_sq"] = frame["discount_depth_model"] ** 2
    frame["price_imputed_indicator"] = frame["price_imputed"].fillna(False).astype(int)
    for column in ("store_upc", "upc", "calendar_month"):
        frame[column] = frame[column].astype("category")
    required = [
        "move", "log_price_model", "promotion_indicator", "post_promotion_indicator",
        "discount_depth_model", "discount_depth_sq", "calendar_month", "scaled_time_trend",
        "thanksgiving_week", "christmas_week", "new_year_week", "easter_week",
        "price_imputed_indicator", "store_upc", "upc", "week", "regular_price",
    ]
    return frame.dropna(subset=required).sort_values(["store_upc", "week"], kind="mergesort").reset_index(drop=True)


def run_alternative_ppml_robustness() -> dict[str, object]:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    main_path = ROOT / "artifacts" / "policy" / "policy_optimization.pkl"
    main = load_pickle(main_path)
    planning = main["schedule_system"]["planning"]
    source_weeks = np.asarray(main["source_weeks"], dtype=int)
    origin = int(source_weeks.min())

    # Dependency audit: behavioral draws were calibrated from residualized raw
    # calibration data, not PPML fitted values. Reuse them byte-for-byte.
    draws = main["draws_by_product"]
    actions = main["action_sets"]
    source = _source_frame(main)
    predictions = common_origin_ex_ante_predictions(
        source, source_weeks, MODEL, planning_origin_week=origin,
        common_controls=POLICY_COMMON_CONTROLS,
    )
    predictions["split"] = "test"
    if not (predictions["prediction_design"].eq("ex_ante_fixed_grid").all() and predictions["origin_predictors_verified"].all()):
        raise AssertionError("Alternative PPML profile failed common-origin provenance assertions.")
    predictions.to_pickle(ARTIFACTS / "common_origin_predictions_product_promotion.pkl")

    # The only changed policy input is the PPML baseline.  Fixed prices/costs,
    # actions, behavioral draws, planning, and reimbursement grid are reused.
    profile_table, profile_raw, selected_weeks = build_weekly_economic_profiles(
        source, predictions, main["products"], planning.decision_horizon,
        planning.washout_horizon, model_name=MODEL,
    )
    if list(selected_weeks) != list(source_weeks):
        raise AssertionError("Alternative profile weeks differ from frozen main horizon.")
    profiles = demand_only_profiles(profile_raw)
    dynamic = build_schedule_system(draws, profiles, actions, planning, main["reimbursement_grid"])
    naive = build_schedule_system(draws, profiles, actions, planning, main["reimbursement_grid"], add_new_promotion_displacement=False)
    run = run_three_policy_grid(dynamic, naive, draws, profiles, actions, main["reimbursement_grid"], main["capacities"], compute_second_best=False)

    summary, full = summarize("alternative_product_promotion", run["results"], run["schedules"], main["capacities"])
    comparison = main["policy_results"][["capacity", "reimbursement_share", "delta_plan", "delta_disp", "delta_total"]].merge(
        full[["capacity", "reimbursement_share", "delta_plan", "delta_disp", "delta_total"]],
        on=["capacity", "reimbursement_share"], suffixes=("_main", "_alternative"), validate="one_to_one",
    )
    for component in ("plan", "disp", "total"):
        comparison[f"delta_{component}_difference_alternative_minus_main"] = comparison[f"delta_{component}_alternative"] - comparison[f"delta_{component}_main"]
    full.to_csv(OUT / "alternative_ppml_full_grid.csv", index=False)
    summary.to_csv(OUT / "alternative_ppml_peak_transition_summary.csv", index=False)
    comparison.to_csv(OUT / "alternative_ppml_vs_main_full_grid.csv", index=False)
    save_pickle({"run": run, "weekly_profiles": profiles, "weekly_profile_table": profile_table,
                 "provenance": {"main_artifact": str(main_path), "ppml_model": MODEL,
                                "behavioral_draws_reused": True, "behavioral_dependency": "independent_residualized_raw_calibration",
                                "source_weeks": list(map(int, source_weeks))}},
                ARTIFACTS / "policy_optimization.pkl")
    return {
        "predictions": predictions,
        "weekly_profile_table": profile_table,
        "run": run,
        "summary": summary,
        "full_grid": full,
        "comparison": comparison,
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    output = run_alternative_ppml_robustness()
    print(output["summary"].to_string(index=False))
    print(f"Completed alternative-PPML robustness in {output['runtime_seconds']:.1f}s.")


if __name__ == "__main__":
    main()
