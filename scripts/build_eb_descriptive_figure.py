"""Create Figure 1 from Notebook-03 support and Notebook-05 EB draws."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: list[float]) -> np.ndarray:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return np.interp(probabilities, np.cumsum(weights) / weights.sum(), values)


def main() -> None:
    support_path = ROOT / "results" / "empirical_bayes" / "tables" / "paper_promotion_depth_support.csv"
    draw_path = ROOT / "artifacts" / "calibration" / "empirical_bayes" / "product_behavioral_draws.pkl"
    lookup_path = ROOT / "data" / "processed" / "cereal_product_lookup.parquet"
    figure_dir = ROOT / "results" / "empirical_bayes" / "figures"
    table_dir = ROOT / "results" / "empirical_bayes" / "tables"
    for path in (support_path, draw_path, lookup_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required upstream artifact is missing: {path}")
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    support = pd.read_csv(support_path)
    draws = pd.read_pickle(draw_path).copy()
    draws["upc"] = draws["upc"].astype(str)
    lookup = pd.read_parquet(lookup_path)
    lookup["upc"] = lookup["upc"].astype(str)
    labels = lookup[["upc", "descrip", "size"]].drop_duplicates("upc").copy()
    labels["product_label"] = labels["descrip"].astype(str).str.strip() + " (" + labels["size"].astype(str).str.strip() + ")"

    rows: list[dict[str, float | str]] = []
    for upc, group in draws.groupby("upc", observed=True):
        weights = group["draw_weight"].to_numpy(float)
        if not np.isclose(weights.sum(), 1.0):
            raise AssertionError(f"EB draw weights do not sum to one for {upc}.")
        state = np.full(len(group), 0.10)
        effects = []
        for _ in range(4):
            effects.append(group["base_demand"].to_numpy(float) * (np.exp(-group["displacement_strength"].to_numpy(float) * state) - 1.0))
            state *= group["inventory_persistence"].to_numpy(float)
        draw_effect = np.mean(np.vstack(effects), axis=0)
        p05, p50, p95 = _weighted_quantile(draw_effect, weights, [0.05, 0.50, 0.95])
        rows.append({"upc": upc, "mean_effect": float(draw_effect @ weights), "median_effect": float(p50), "effect_p05": float(p05), "effect_p95": float(p95)})
    effects = pd.DataFrame(rows).merge(labels[["upc", "product_label"]], on="upc", how="left", validate="one_to_one").sort_values("mean_effect", ascending=False)
    if effects["product_label"].isna().any():
        raise AssertionError("Missing product labels for EB Figure 1(b).")
    effects.to_csv(table_dir / "paper_model_implied_post_promotion_effects.csv", index=False)

    depth = support.groupby("depth_cluster", observed=True).agg(observations=("observations", "sum")).reset_index().sort_values("depth_cluster")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), constrained_layout=True)
    depth_percent = 100.0 * depth["depth_cluster"]
    axes[0].bar(depth_percent, depth["observations"], width=4.0)
    axes[0].set(xlabel="Promotion depth (%)", ylabel="Product-store-week observations", title="(a) Observed promotion depths")
    axes[0].set_xticks(np.arange(int(5 * np.floor(depth_percent.min() / 5)), int(5 * np.ceil(depth_percent.max() / 5)) + 1, 5))
    axes[0].text(0.98, 0.95, rf"$N={int(depth['observations'].sum()):,}$", transform=axes[0].transAxes, ha="right", va="top")
    plot = effects.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    axes[1].hlines(y, plot["effect_p05"], plot["effect_p95"], color="#8db9d8", linewidth=2.3)
    axes[1].scatter(plot["mean_effect"], y, color="#1f77b4", s=40, zorder=2)
    axes[1].axvline(0.0, color="#1f77b4", linestyle=":", linewidth=1.0)
    axes[1].set(yticks=y, yticklabels=plot["product_label"], xlabel="Predicted change in weekly sales, post-promotion weeks 1–4", title="(b) Model-implied post-promotion effects")
    # Use ASCII punctuation so the label renders consistently on Windows.
    axes[1].set_xlabel("Predicted change in weekly sales, post-promotion weeks 1-4")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"paper_promotion_depths_and_model_implied_post_effects.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    print("Saved EB Figure 1:", figure_dir)


if __name__ == "__main__":
    main()
