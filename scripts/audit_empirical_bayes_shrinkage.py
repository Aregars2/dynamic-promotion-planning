"""Pre-rerun diagnostics for parameter-specific empirical-Bayes pooling."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


PARAMETERS = {
    "price_elasticity": ("price_elasticity", 0.10, 5.00),
    "promotion_lift_log": ("promotion_lift_log", -1.00, 3.00),
    "displacement_strength": ("post1_depth_slope", 0.00, 3.00),
}


def _old_reliability() -> pd.Series:
    draws = pd.read_pickle(ROOT / "artifacts" / "calibration" / "product_behavioral_draws.pkl")
    draws["upc"] = draws["upc"].astype(str)
    return draws.groupby("upc", observed=True)["shrinkage_reliability"].first()


def main() -> None:
    out = ROOT / "results" / "empirical_bayes_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    bootstrap = pd.read_pickle(ROOT / "artifacts" / "calibration" / "product_behavioral_bootstrap.pkl").copy()
    bootstrap["upc"] = bootstrap["upc"].astype(str)
    products = sorted(bootstrap["upc"].unique())
    old = _old_reliability()
    rows, aggregate_rows, sensitivity_rows = [], [], []

    for name, (column, lower, upper) in PARAMETERS.items():
        product_statistics = []
        for upc in products:
            values = bootstrap.loc[bootstrap["upc"].eq(upc), column].to_numpy(dtype=float)
            # This is the exact policy-scale transformation before EB moments.
            values = np.clip(values[np.isfinite(values)], lower, upper)
            if len(values) != 1000:
                # A product with no product bootstrap is full-pooling by construction.
                mean, variance, standard_error = np.nan, np.nan, np.nan
            else:
                mean = float(values.mean())
                variance = float(values.var(ddof=1))
                standard_error = float(np.sqrt(variance))
            product_statistics.append((upc, mean, variance, standard_error))
        stats = pd.DataFrame(product_statistics, columns=["upc", "theta_hat", "bootstrap_variance", "bootstrap_sd"])
        observed = stats["theta_hat"].dropna()
        # The requested K=8 MoM uses the eight policy products.  A product with
        # no unpooled estimate is marked forced full pooling and contributes the
        # pooled value, whose between-product deviation is zero by definition.
        if len(observed) != len(products):
            # Do not silently fabricate an unpooled point estimate for fallback.
            tau2 = np.nan
            between = np.nan
            mean_within = np.nan
            pathological = True
            reason = "missing_unpooled_bootstrap_product"
        else:
            between = float(observed.var(ddof=1))
            mean_within = float(stats["bootstrap_variance"].mean())
            tau2 = max(0.0, between - mean_within)
            pathological = bool(np.isclose(tau2, 0.0))
            reason = "tau2_zero" if pathological else "ok"
        for row in stats.itertuples(index=False):
            weight = 0.0 if not np.isfinite(tau2) else tau2 / (tau2 + row.bootstrap_variance) if (tau2 + row.bootstrap_variance) > 0 else 0.0
            rows.append({"parameter": name, "upc": row.upc, "theta_hat": row.theta_hat,
                         "bootstrap_sd": row.bootstrap_sd, "bootstrap_variance": row.bootstrap_variance,
                         "tau2": tau2, "eb_weight": weight, "old_heuristic_R": float(old.loc[row.upc]),
                         "eb_minus_old_R": weight - float(old.loc[row.upc]), "policy_transformation": f"clip[{lower}, {upper}]"})
            for scale in (0.5, 1.0, 2.0):
                scaled_tau2 = np.nan if not np.isfinite(tau2) else scale * tau2
                scaled_weight = 0.0 if not np.isfinite(scaled_tau2) else scaled_tau2 / (scaled_tau2 + row.bootstrap_variance) if (scaled_tau2 + row.bootstrap_variance) > 0 else 0.0
                sensitivity_rows.append({"parameter": name, "upc": row.upc, "tau2_scale": scale, "tau2": scaled_tau2, "eb_weight": scaled_weight})
        weights = [entry["eb_weight"] for entry in rows if entry["parameter"] == name]
        aggregate_rows.append({"parameter": name, "policy_transformation": f"clip[{lower}, {upper}]",
                               "products": len(products), "observed_cross_product_variance_S2": between,
                               "mean_within_product_bootstrap_variance": mean_within, "tau2": tau2,
                               "tau2_hits_zero": bool(np.isfinite(tau2) and np.isclose(tau2, 0.0)),
                               "pathological": pathological, "diagnostic_status": reason,
                               "minimum_eb_weight": float(np.nanmin(weights)), "median_eb_weight": float(np.nanmedian(weights)), "maximum_eb_weight": float(np.nanmax(weights))})
    detail = pd.DataFrame(rows)
    aggregate = pd.DataFrame(aggregate_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    detail.to_csv(out / "empirical_bayes_weight_diagnostics.csv", index=False)
    aggregate.to_csv(out / "empirical_bayes_variance_summary.csv", index=False)
    sensitivity.to_csv(out / "empirical_bayes_tau2_sensitivity_weights.csv", index=False)
    print(aggregate.to_string(index=False))
    if aggregate["pathological"].any():
        raise RuntimeError("EB diagnostics are not yet admissible for a policy rerun; inspect exported diagnostics.")


if __name__ == "__main__":
    main()
