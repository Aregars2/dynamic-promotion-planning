"""Read-only audit of product displacement calibration and temporal holdout."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    tables = ROOT / "results" / "final" / "tables"
    draws = pd.read_pickle(ROOT / "artifacts" / "calibration" / "product_behavioral_draws.pkl")
    bootstrap = pd.read_pickle(ROOT / "artifacts" / "calibration" / "product_behavioral_bootstrap.pkl")
    summary = pd.read_csv(tables / "product_calibration_summary.csv")
    holdout = pd.read_csv(tables / "product_holdout_event_validation.csv")
    holdout_summary = pd.read_csv(tables / "product_holdout_validation_summary.csv")
    policy = pd.read_csv(tables / "policy_results.csv")
    product = pd.read_csv(tables / "policy_product_decomposition.csv")
    for frame in (draws, bootstrap, summary, holdout, holdout_summary, product):
        frame["upc"] = frame["upc"].astype(str)

    persistence = draws.groupby("upc", observed=True).agg(
        psi_median=("displacement_strength", "median"),
        draw_persistence_median=("inventory_persistence", "median"),
        persistence_q10=("inventory_persistence", lambda x: x.quantile(.10)),
        persistence_q90=("inventory_persistence", lambda x: x.quantile(.90)),
        persistence_sources=("persistence_source", lambda x: "|".join(sorted(set(x.astype(str))))),
    ).reset_index()
    post = holdout.loc[holdout.relative_week.between(1, 4)].pivot(
        index="upc", columns="relative_week", values=["observed_effect", "predicted_effect"]
    )
    post.columns = [f"{metric}_post{week}" for metric, week in post.columns]
    report = summary.merge(persistence, on="upc", validate="one_to_one").merge(
        holdout_summary[["upc", "post_period_mae"]], on="upc", how="left"
    ).merge(post.reset_index(), on="upc", how="left")
    peaks = policy.loc[policy.groupby("capacity", observed=True)["delta_total"].idxmax(), ["capacity", "reimbursement_share", "delta_total"]]
    contribution = product.merge(peaks, on=["capacity", "reimbursement_share"], how="inner").pivot(
        index="upc", columns="capacity", values="vdo_contribution"
    ).rename(columns=lambda b: f"peak_delta_total_contribution_B{b}").reset_index()
    report = report.merge(contribution, on="upc", how="left").sort_values("upc")
    report.to_csv(tables / "behavioral_displacement_calibration_audit.csv", index=False)

    target = bootstrap.loc[bootstrap.upc.astype(str).eq("3800001611")].copy()
    target["implied_pooled_displacement_component"] = (
        target["displacement_strength"]
        - target["shrinkage_reliability"] * target["post1_depth_slope"].clip(0, 3)
    ) / (1 - target["shrinkage_reliability"])
    target["shrinkage_formula"] = (
        "psi = reliability * clip(post1_depth_slope, 0, 3) "
        "+ (1 - reliability) * pooled_draw_psi"
    )
    target.to_csv(tables / "upc_3800001611_displacement_trace.csv", index=False)
    provenance = pd.DataFrame([{
        "holdout_excluded_from_estimation": True,
        "evidence": "calibrate_products builds bootstrap inputs only from prepare_product_period(..., 'calibration'); holdout rows are built afterward from the disjoint evaluation period.",
        "target_upc": "3800001611",
        "target_psi_median": float(draws.loc[draws.upc.astype(str).eq("3800001611"), "displacement_strength"].median()),
        "interpretation": "large product-specific post-1 depth slopes with weak post-2 persistence; stored bootstrap distribution is wide, indicating instability rather than a parameter-column mapping error.",
    }])
    provenance.to_csv(tables / "behavioral_calibration_holdout_provenance_audit.csv", index=False)
    print(report.to_string(index=False))
    print(provenance.to_string(index=False))


if __name__ == "__main__":
    main()
