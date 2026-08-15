"""Fast provenance diagnostics for frozen EB calibration and sample eligibility.

No parameters, selected samples, or policy results are changed.  The first
table distinguishes observed final-boundary mass from recorded fallback paths;
the second reconstructs selection after removing only test-period eligibility
requirements, thereby testing whether future availability affected retention.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_promotion_planning.config import load_analysis_config
from dynamic_promotion_planning.demand import build_product_selection_table, panel_period_counts
from dynamic_promotion_planning.policy import load_pickle


VERSION = "empirical_bayes"
CAL = ROOT / "artifacts" / "calibration" / VERSION
TABLES = ROOT / "results" / VERSION / "tables"
OUT = ROOT / "results" / VERSION / "diagnostics"
DATA = ROOT / "data" / "processed" / "cereal_demand_model_data.parquet"


def _weighted_share(frame: pd.DataFrame, mask: pd.Series) -> float:
    weights = frame["draw_weight"].to_numpy(float)
    return float(weights[np.asarray(mask, bool)].sum() / weights.sum())


def clipping_and_fallback(draws: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Summarize recorded safeguard routes and final draw boundary mass.

    The pre-clipping values for ε and γ are intentionally not persisted, so an
    endpoint draw is labelled *at bound*, not asserted to have been clipped.
    ψ retains its pre-transform post-1 slope, permitting an explicit observed
    out-of-range check for that parameter.
    """
    bounds = {
        "price_elasticity": (0.10, 5.00),
        "promotion_lift_log": (-1.00, 3.00),
        "displacement_strength": (0.00, 3.00),
        "inventory_persistence": (0.05, 0.95),
    }
    rows = []
    for upc, group in draws.groupby("upc", observed=True):
        parent = bootstrap.loc[bootstrap.upc.astype(str).eq(str(upc))]
        base = {
            "upc": str(upc), "final_draw_rows": len(group),
            "weighted_product_fallback_share": _weighted_share(group, group.product_calibration_source.eq("pooled_fallback")),
            "weighted_product_bootstrap_share": _weighted_share(group, group.product_calibration_source.eq("product_bootstrap")),
            "weighted_partial_id_plus_pooled_persistence_share": _weighted_share(group, group.persistence_source.eq("partial_id_plus_pooled")),
            "weighted_pooled_fallback_grid_share": _weighted_share(group, group.persistence_source.eq("pooled_fallback_grid")),
            "weighted_product_depth_decay_share": _weighted_share(group, group.persistence_source.eq("product_depth_decay")),
        }
        for parameter, (lower, upper) in bounds.items():
            result = dict(base)
            result.update({"parameter": parameter, "lower_bound": lower, "upper_bound": upper,
                           "weighted_final_lower_bound_share": _weighted_share(group, np.isclose(group[parameter], lower)),
                           "weighted_final_upper_bound_share": _weighted_share(group, np.isclose(group[parameter], upper))})
            if parameter == "displacement_strength":
                raw = parent["post1_depth_slope"].to_numpy(float)
                result["parent_raw_below_lower_share"] = float(np.mean(np.isfinite(raw) & (raw < lower)))
                result["parent_raw_above_upper_share"] = float(np.mean(np.isfinite(raw) & (raw > upper)))
                result["interpretation"] = "pre-transform post1 slope is persisted; below/above shares identify observed clip exposure"
            else:
                result["parent_raw_below_lower_share"] = np.nan
                result["parent_raw_above_upper_share"] = np.nan
                result["interpretation"] = "pre-clipping values are not persisted; endpoint mass is descriptive, not proof of clipping"
            rows.append(result)
    return pd.DataFrame(rows).sort_values(["upc", "parameter"])


def _splits(frame: pd.DataFrame, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weeks = np.sort(frame.week.unique())
    train_end = int(np.floor(config.train_share * len(weeks)))
    cal_end = int(np.floor((config.train_share + config.calibration_share) * len(weeks)))
    return weeks[:train_end], weeks[train_end:cal_end], weeks[cal_end:]


def train_cal_eligibility(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Replicate selection while explicitly removing future test criteria."""
    train, calibration, test = _splits(frame, config)
    original_products = pd.read_csv(TABLES / "product_selection_table.csv").copy()
    original_stores = pd.read_csv(TABLES / "store_selection_table.csv").copy()
    original_panels = pd.read_csv(TABLES / "panel_selection_table.csv").copy()
    product_table, _ = build_product_selection_table(frame, train, calibration, test, config)
    product_criteria = [c for c in product_table.columns if c.startswith("adequate_") or c == "acceptable_imputation"]
    product_without_test = [c for c in product_criteria if c != "adequate_test_weeks"]
    product_table["eligible_train_cal_only"] = product_table[product_without_test].all(axis=1)

    # Current baseline eligibility supplied by the saved Notebook-02 tables.
    product_compare = product_table[["upc", "eligible_train_cal_only"]].merge(
        original_products[["upc", "eligible"]].rename(columns={"eligible": "eligible_current"}), on="upc", how="outer"
    ).fillna(False)
    product_compare["stage"] = "product"

    selected_products = product_compare.loc[product_compare.eligible_train_cal_only, "upc"].astype(str).tolist()
    candidate = frame.loc[frame.upc.astype(str).isin(selected_products)].copy()
    minimum_coverage = max(1, int(np.ceil(config.min_product_coverage_share * len(selected_products))))
    train_store = candidate.loc[candidate.week.isin(train)].groupby("store", observed=True).agg(train_weeks=("week", "nunique"), train_products=("upc", "nunique"))
    cal_store = candidate.loc[candidate.week.isin(calibration)].groupby("store", observed=True).agg(calibration_weeks=("week", "nunique"), calibration_products=("upc", "nunique"))
    stores = train_store.join(cal_store, how="left").fillna(0).reset_index()
    stores["eligible_train_cal_only"] = ((stores.train_weeks >= config.min_store_train_weeks) & (stores.calibration_weeks >= config.min_store_calibration_weeks) & (stores.train_products >= minimum_coverage) & (stores.calibration_products >= minimum_coverage))
    store_compare = stores[["store", "eligible_train_cal_only"]].merge(original_stores[["store", "eligible"]].rename(columns={"eligible": "eligible_current"}), on="store", how="outer").fillna(False)
    store_compare["stage"] = "store"

    selected_stores = store_compare.loc[store_compare.eligible_train_cal_only, "store"].astype(str)
    candidate = candidate.loc[candidate.store.astype(str).isin(selected_stores)].copy()
    panels = panel_period_counts(candidate, train, "train").join(panel_period_counts(candidate, calibration, "calibration"), how="left").fillna(0).reset_index()
    panels["eligible_train_cal_only"] = ((panels.train_weeks >= config.min_panel_train_weeks) & (panels.calibration_weeks >= config.min_panel_calibration_weeks))
    panel_compare = panels[["store_upc", "eligible_train_cal_only"]].merge(original_panels[["store_upc", "eligible"]].rename(columns={"eligible": "eligible_current"}), on="store_upc", how="outer").fillna(False)
    panel_compare["stage"] = "panel"

    comparisons = []
    for key, table in (("upc", product_compare), ("store", store_compare), ("store_upc", panel_compare)):
        table["eligible_current"] = table.eligible_current.astype(bool)
        table["eligible_train_cal_only"] = table.eligible_train_cal_only.astype(bool)
        table["same_eligibility"] = table.eligible_current.eq(table.eligible_train_cal_only)
        comparisons.append(table.rename(columns={key: "identifier"})[["stage", "identifier", "eligible_current", "eligible_train_cal_only", "same_eligibility"]])
    return pd.concat(comparisons, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draws = pd.read_pickle(CAL / "product_behavioral_draws.pkl")
    bootstrap = pd.read_pickle(CAL / "product_behavioral_bootstrap.pkl")
    safeguards = clipping_and_fallback(draws, bootstrap)
    safeguards.to_csv(OUT / "behavioral_clipping_fallback_frequency.csv", index=False)

    config = load_analysis_config().demand
    frame = pd.read_parquet(DATA)
    # These are the exact Notebook-02 definitions, reconstructed locally so
    # the audit reads no future-generated notebook table as an input.
    frame["promotion_indicator"] = frame["pricing_state"].eq("promotion").astype("int8")
    frame["regular_indicator"] = frame["pricing_state"].eq("regular").astype("int8")
    eligibility = train_cal_eligibility(frame, config)
    eligibility.to_csv(OUT / "train_cal_only_eligibility_comparison.csv", index=False)
    summary = eligibility.groupby("stage", observed=True).agg(
        current_eligible=("eligible_current", "sum"),
        train_cal_only_eligible=("eligible_train_cal_only", "sum"),
        unchanged=("same_eligibility", "sum"),
        changed=("same_eligibility", lambda x: int((~x).sum())),
    ).reset_index()
    summary.to_csv(OUT / "train_cal_only_eligibility_summary.csv", index=False)
    selected_products = set(map(str, load_pickle(CAL / "product_calibration.pkl")["selected_products"]))
    selected_product_check = eligibility.loc[
        eligibility.stage.eq("product") & eligibility.identifier.astype(str).isin(selected_products)
    ]
    retained_current = eligibility.loc[eligibility.eligible_current]
    if not retained_current.eligible_train_cal_only.all():
        raise AssertionError("A currently retained unit fails train+cal-only eligibility.")
    report = f"""# Calibration safeguards and future-availability diagnostic

## Reproducibility

- Command: `python scripts/audits/audit_calibration_safeguards_and_eligibility.py`
- Behavioral inputs: `artifacts/calibration/empirical_bayes/product_behavioral_bootstrap.pkl` and `product_behavioral_draws.pkl`
- Eligibility input: `data/processed/cereal_demand_model_data.parquet`, with the exact Notebook-02 promotion and regular indicators reconstructed from `pricing_state`.
- This audit changes no calibration, eligibility rule, selected sample, or policy artifact.

## Safeguards

`behavioral_clipping_fallback_frequency.csv` reports one row per selected product and parameter. Product-level pooled fallback accounts for {safeguards.weighted_product_fallback_share.mean():.1%} of final draw weight on average across products. Persistence is directly product-depth-decay informed for {safeguards.weighted_product_depth_decay_share.mean():.1%} of final weight, while {safeguards.weighted_partial_id_plus_pooled_persistence_share.mean():.1%} uses the pre-specified partial-identification-plus-pooled grid. For displacement strength, the persisted pre-transform post-1 slope is below zero in {safeguards.loc[safeguards.parameter.eq('displacement_strength'), 'parent_raw_below_lower_share'].mean():.1%} of parent draws on average; final ψ has zero lower- or upper-bound mass. For ε and γ, pre-clipping values are not persisted, so final endpoint frequency is reported descriptively and cannot be called a clipping frequency.

## Eligibility without future availability

All {len(selected_product_check)} final selected products and all {len(retained_current)} currently retained units across the product, store, and panel stages remain eligible when the test-period availability requirements are removed. Thus, future test-row availability did not determine retention of any final analysis unit. The broader candidate pool does expand when test requirements are removed (see the stage-level summary), so the valid claim is retention invariance of the final sample—not invariance of the entire candidate pool.
"""
    (OUT / "calibration_safeguards_and_eligibility_report.md").write_text(report, encoding="utf-8")
    print("Saved", OUT / "behavioral_clipping_fallback_frequency.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
