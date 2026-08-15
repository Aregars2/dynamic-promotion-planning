"""Create a separate 48-week same-horizon normalization version of Figure 2."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = "empirical_bayes_price_consistent"
GRID = ROOT / "results" / VERSION / "robustness" / "same_horizon_normalization_full_grid.csv"
OUT = ROOT / "results" / VERSION / "figures"


def main() -> None:
    if not GRID.is_file():
        raise FileNotFoundError(f"Missing 48-week normalization grid: {GRID}. Run final robustness reporting first.")
    frame = pd.read_csv(GRID).loc[lambda x: x.reimbursement_share.between(0.50, 1.00)].copy()
    required = [
        "delta_plan_pct_myopic_same_horizon_48w",
        "delta_disp_pct_myopic_same_horizon_48w",
        "delta_total_pct_myopic_same_horizon_48w",
    ]
    if frame.empty or frame[required].isna().any().any():
        raise AssertionError("48-week normalization grid is incomplete in the requested plot range.")
    if not np.allclose(frame[required[0]] + frame[required[1]], frame[required[2]], atol=1e-10):
        raise AssertionError("48-week normalized components do not add.")
    OUT.mkdir(parents=True, exist_ok=True)
    components = [
        (required[0], r"$\Delta^{\mathrm{plan}}$", "#6494EDD3", "--", 1.7),
        (required[1], r"$\Delta^{\mathrm{disp}}$", "coral", "-.", 1.7),
        (required[2], r"$\Delta^{\mathrm{total}}$", "#022278D2", "-", 2.8),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True, sharey=True)
    for ax, capacity in zip(axes.flat, [1, 2, 3, 8], strict=True):
        group = frame.loc[frame.capacity.eq(capacity)].sort_values("reimbursement_share")
        for column, _, color, linestyle, linewidth in components:
            ax.plot(group.reimbursement_share, group[column], color=color, linestyle=linestyle, linewidth=linewidth)
        # Peak location is defined on absolute Δtotal; the plotted height is
        # subsequently normalized by 48-week πM profit.
        peak = group.loc[group["delta_total"].idxmax()]
        ax.scatter(peak.reimbursement_share, peak[required[2]], s=48, color="#022278D2", edgecolor="white", linewidth=1.0, zorder=5)
        ax.axhline(0, color="0.45", linewidth=0.55, linestyle=":")
        ax.set_title(f"B={capacity}", pad=12)
        ax.set_xlim(0.50, 1.00)
        ax.set_xticks(np.arange(0.50, 1.01, 0.10))
        ax.grid(axis="x", color="0.90", linewidth=0.8)
    ymax = max(0.1, float(frame[required].max().max()))
    ymin = min(-0.05, float(frame[required].min().min()))
    for ax in axes.flat:
        ax.set_ylim(min(-0.05, np.floor(ymin * 10) / 10), np.ceil((ymax + 0.03) * 10) / 10)
    fig.supxlabel(r"Reimbursement share, $\lambda$", y=0.04)
    fig.supylabel("Value contrast (% of 48-week myopic profit)", x=0.03)
    handles = [Line2D([0], [0], color=c, linestyle=s, linewidth=w, label=label) for _, label, c, s, w in components]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#022278D2", markeredgecolor="white", markersize=8, label="Absolute-total peak"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.11, hspace=0.30, wspace=0.13)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"three_policy_components_by_capacity_lambda_050_to_100_48_week_normalized.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)
    print("Saved 48-week-normalized Figure 2 to", OUT)


if __name__ == "__main__":
    main()
