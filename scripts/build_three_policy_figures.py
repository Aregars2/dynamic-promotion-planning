"""Create paper-facing three-policy figures and transition table.

Run after ``06_policy_optimization.ipynb``.  The optimizer is intentionally
not invoked here: this script only validates and reports its saved outputs.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

# Reports are generated in batch environments without a GUI/Tk installation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dynamic_promotion_planning.boundaries import (
    _calendar_hash,
    build_three_policy_transition_table,
    compare_schedule_maps,
)
from dynamic_promotion_planning.policy import load_pickle


def _require_fine_candidate_grid(artifact: dict) -> None:
    """Reject an artifact whose pruned candidates were not built on 0.01 grid."""
    candidate_grid = np.asarray(artifact["schedule_system"]["alpha_grid"], dtype=float)
    required_grid = np.round(np.arange(0.0, 1.001, 0.01), 8)
    if not np.array_equal(np.round(candidate_grid, 8), required_grid):
        raise AssertionError(
            "Candidate schedules were not pruned on the required 0.01 reimbursement "
            "grid. Re-run Notebook 06 after updating config/analysis.toml."
        )


def _three_policy_results(artifact: dict) -> pd.DataFrame:
    results = artifact["policy_results"].copy()
    required = {"value_piM", "value_piN", "value_piD", "delta_plan", "delta_disp", "delta_total"}
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(
            "The policy artifact lacks the requested three-policy outputs: "
            f"{sorted(missing)}. Re-run Notebook 06."
        )
    if not np.allclose(results["delta_plan"] + results["delta_disp"], results["delta_total"], atol=1e-8):
        raise AssertionError("Δplan + Δdisp != Δtotal in the saved policy results.")
    return results.sort_values(["capacity", "reimbursement_share"]).reset_index(drop=True)


def _switch_points(frame: pd.DataFrame, signature_column: str) -> pd.DataFrame:
    output = frame.copy()
    output["switch"] = output[signature_column].ne(output[signature_column].shift())
    if not output.empty:
        output.loc[output.index[0], "switch"] = False
    return output.loc[output["switch"]]


def _normalized_components(results: pd.DataFrame, task8_grid: pd.DataFrame) -> pd.DataFrame:
    """Attach Task-8 12-week-myopic-profit percentages to calendar results."""
    percentage_columns = ["delta_plan_pct", "delta_disp_pct", "delta_total_pct"]
    main_grid = task8_grid.loc[
        task8_grid["specification"].eq("main"),
        ["capacity", "reimbursement_share", "myopic_planning_profit", *percentage_columns],
    ].copy()
    output = results.merge(
        main_grid,
        on=["capacity", "reimbursement_share"],
        how="left",
        validate="one_to_one",
    )
    if output[percentage_columns].isna().any().any():
        raise AssertionError("Task-8 main reporting grid is incomplete for the policy results.")
    if np.any(np.isclose(output["myopic_planning_profit"], 0.0)):
        raise AssertionError("Cannot normalize policy components by zero 12-week myopic profit.")
    for source, destination in zip(percentage_columns, (
        "delta_plan_percent_myopic", "delta_disp_percent_myopic", "delta_total_percent_myopic",
    )):
        output[destination] = output[source]
    if not np.allclose(
        output["delta_plan_percent_myopic"] + output["delta_disp_percent_myopic"],
        output["delta_total_percent_myopic"],
        atol=1e-8,
    ):
        raise AssertionError("Normalized value components do not add up.")
    return output


def _calendar_audit(results: pd.DataFrame, schedules: dict) -> pd.DataFrame:
    """Record the low-share identity check and any negative planning components."""
    rows: list[dict[str, object]] = []
    signatures = [
        "myopic_schedule_signature",
        "naive_dynamic_schedule_signature",
        "dynamic_schedule_signature",
    ]
    for capacity, group in results.groupby("capacity", observed=True):
        low = group.loc[group["reimbursement_share"] < 0.80]
        identical = low[signatures].nunique(axis=1).eq(1).all()
        zero_components = low[["delta_plan", "delta_disp", "delta_total"]].abs().max(axis=1).le(1e-9).all()
        rows.append({
            "audit": "low_reimbursement_identity",
            "capacity": int(capacity),
            "reimbursement_share": np.nan,
            "finding": "identical calendars and zero components below lambda=0.80",
            "grid_points": len(low),
            "calendar_identity_verified": bool(identical),
            "zero_components_verified": bool(zero_components),
        })

    for row in results.loc[results["delta_plan"] < -1e-9].itertuples(index=False):
        key = (round(float(row.reimbursement_share), 8), int(row.capacity))
        policy_schedules = schedules[key]
        comparison = compare_schedule_maps(
            policy_schedules["myopic"], policy_schedules["naive_dynamic"]
        )
        rows.append({
            "audit": "negative_forward_planning_component",
            "capacity": int(row.capacity),
            "reimbursement_share": float(row.reimbursement_share),
            "finding": "piN is evaluated below piM under common full-displacement replay",
            "grid_points": np.nan,
            "calendar_identity_verified": False,
            "zero_components_verified": False,
            "value_piM": float(row.value_piM),
            "value_piN": float(row.value_piN),
            "value_piD": float(row.value_piD),
            "delta_plan": float(row.delta_plan),
            "delta_disp": float(row.delta_disp),
            "delta_total": float(row.delta_total),
            "piM_calendar_hash": _calendar_hash(policy_schedules["myopic"]),
            "piN_calendar_hash": _calendar_hash(policy_schedules["naive_dynamic"]),
            "changed_product_week_decisions": comparison["changed_cells"],
            "changed_products": comparison["changed_products"],
        })
    return pd.DataFrame(rows)


def _cross_capacity_transition_summary(normalized: pd.DataFrame) -> pd.DataFrame:
    """Summarize calendar switching and peak gains without privileging one B."""
    signatures = {
        "piM_switches": "myopic_schedule_signature",
        "piN_switches": "naive_dynamic_schedule_signature",
        "piD_switches": "dynamic_schedule_signature",
    }
    rows: list[dict[str, object]] = []
    for capacity, group in normalized.groupby("capacity", observed=True):
        group = group.sort_values("reimbursement_share").reset_index(drop=True)
        peak = group.loc[group["delta_total_percent_myopic"].idxmax()]
        switch_counts = {
            name: int(group[column].ne(group[column].shift()).iloc[1:].sum())
            for name, column in signatures.items()
        }
        total_at_peak = float(peak["delta_total_percent_myopic"])
        rows.append({
            "capacity": int(capacity),
            **switch_counts,
            "max_total_gain_percent_myopic": total_at_peak,
            "reimbursement_share_at_max_total_gain": float(peak["reimbursement_share"]),
            "displacement_share_at_peak_percent": (
                100.0 * float(peak["delta_disp_percent_myopic"]) / total_at_peak
                if not np.isclose(total_at_peak, 0.0)
                else np.nan
            ),
            "mean_dynamic_binding_weeks_lambda_ge_080": float(
                group.loc[group["reimbursement_share"].ge(0.80), "dynamic_binding_weeks"].mean()
            ),
        })
    return pd.DataFrame(rows).sort_values("capacity").reset_index(drop=True)


def _write_switching_rug(
    group: pd.DataFrame,
    capacity: int,
    figure_dir: Path,
) -> None:
    """Create a compact, qualitative calendar-switch rug for one capacity."""
    group = group.sort_values("reimbursement_share").reset_index(drop=True)
    peak = group.loc[group["delta_total_percent_myopic"].idxmax()]
    fig, ax = plt.subplots(figsize=(8.2, 1.60))

    ax.axvline(
        peak["reimbursement_share"], color="0.55", linewidth=0.65,
        linestyle="--", zorder=0,
    )
    signatures = [
        ("myopic_schedule_signature", 2),
        ("naive_dynamic_schedule_signature", 1),
        ("dynamic_schedule_signature", 0),
    ]
    for signature, row in signatures:
        points = _switch_points(group, signature)
        if not points.empty:
            ax.vlines(
                points["reimbursement_share"], row - 0.21, row + 0.21,
                color="0.20", linewidth=1.15, zorder=2,
            )
    ax.set_xlim(0.75, 1.00)
    ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticks([2, 1, 0], [r"$\pi^M$", r"$\pi^N$", r"$\pi^D$"])
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel(r"Reimbursement share, $\lambda$")
    ax.grid(axis="x", color="0.90", linewidth=0.7)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.34)
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"three_policy_switching_rug_B{capacity}.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def main() -> None:
    artifact_path = (
        PROJECT_ROOT / "artifacts" / "policy" / "empirical_bayes_price_consistent"
        / "policy_optimization.pkl"
    )
    figure_dir = PROJECT_ROOT / "results" / "empirical_bayes_price_consistent" / "figures"
    table_dir = PROJECT_ROOT / "results" / "empirical_bayes_price_consistent" / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_pickle(artifact_path)
    _require_fine_candidate_grid(artifact)
    results = _three_policy_results(artifact)
    task8_grid_path = table_dir / "task8_three_policy_reporting_grid.csv"
    if not task8_grid_path.exists():
        raise FileNotFoundError(
            "Task-8 reporting grid is required for paper-facing percentages. "
            "Run scripts/run_task8_reporting_checks.py first."
        )
    normalized = _normalized_components(results, pd.read_csv(task8_grid_path))
    schedules = artifact["three_policy_schedules"]

    components = [
        ("delta_plan_percent_myopic", r"$\Delta^{\mathrm{plan}}$"),
        ("delta_disp_percent_myopic", r"$\Delta^{\mathrm{disp}}$"),
        ("delta_total_percent_myopic", r"$\Delta^{\mathrm{total}}$"),
    ]
    # Muted light blue / coral / navy: Total remains visually dominant while
    # retaining distinct line styles for grayscale reproduction.
    colors = ["#6494EDD3", "coral", "#022278D2"]
    color_by_component = dict(zip([column for column, _ in components], colors))
    style_by_component = {
        "delta_plan_percent_myopic": ("--", 1.7),
        "delta_disp_percent_myopic": ("-.", 1.7),
        "delta_total_percent_myopic": ("-", 2.8),
    }
    zoomed = normalized.loc[normalized["reimbursement_share"].ge(0.75)].copy()

    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 7.6), sharex=True, sharey=True
    )
    total_column = "delta_total_percent_myopic"
    for ax, capacity in zip(axes.flat, [1, 2, 3, 8]):
        group = zoomed.loc[zoomed["capacity"].eq(capacity)].sort_values(
            "reimbursement_share"
        )
        if group.empty:
            raise RuntimeError(f"Missing B={capacity} from policy results.")
        for column, _label in components:
            linestyle, linewidth = style_by_component[column]
            ax.plot(
                group["reimbursement_share"],
                group[column],
                color=color_by_component[column], linestyle=linestyle,
                linewidth=linewidth,
            )
        # Define the peak before normalization: the marker denotes the λ
        # maximizing absolute Δtotal, not the maximum percentage.
        peak = group.loc[group["delta_total"].idxmax()]
        ax.scatter(
            peak["reimbursement_share"], peak[total_column],
            s=48, color=color_by_component[total_column], edgecolor="white",
            linewidth=1.0, zorder=5,
        )
        ax.axhline(0, color="0.45", linewidth=0.55, linestyle=":")
        ax.set_title(f"B={capacity}", pad=12)
        ax.set_xlim(0.75, 1.00)
        ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_ylim(-0.5, 2.5)
        ax.grid(axis="x", color="0.9", linewidth=0.8)

    fig.supxlabel(r"Reimbursement share, $\lambda$", y=0.04)
    fig.supylabel(
        "Value contrast (% of 12-week myopic planning profit)", x=0.03
    )

    legend_handles = [
        Line2D(
            [0], [0], color=color_by_component[column],
            linestyle=style_by_component[column][0],
            linewidth=style_by_component[column][1], label=label,
        )
        for column, label in components
    ]
    legend_handles.append(
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=color_by_component[total_column], markeredgecolor="white",
            markersize=8, label="Absolute-total peak",
        )
    )
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.11, hspace=0.30, wspace=0.13)
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"three_policy_components_by_capacity.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)

    transition_tables = []
    for capacity in [1, 2, 3, 8]:
        transition_rows = build_three_policy_transition_table(results, schedules, capacity=capacity)
        transition_rows["capacity"] = capacity
        transition_tables.append(transition_rows)
        if capacity == 2:
            transition_rows.drop(columns="capacity").to_csv(
                table_dir / "three_policy_transition_table_B2.csv", index=False
            )
    _write_switching_rug(
        normalized.loc[normalized["capacity"].eq(2)].copy(), 2, figure_dir
    )
    pd.concat(transition_tables, ignore_index=True).to_csv(
        table_dir / "three_policy_transition_table_all_capacities.csv", index=False
    )
    _cross_capacity_transition_summary(normalized).to_csv(
        table_dir / "three_policy_transition_summary.csv", index=False
    )
    _calendar_audit(results, schedules).to_csv(
        table_dir / "three_policy_calendar_audit.csv", index=False
    )


if __name__ == "__main__":
    main()
