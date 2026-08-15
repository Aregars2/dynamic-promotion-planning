"""Create deterministic LaTeX tables from canonical paper-output CSV files.

This module is presentation glue only: it never refits a model, recalibrates
behavioral draws, or reoptimizes policy calendars.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VERSION = "empirical_bayes_price_consistent"
TABLE_DIR = ROOT / "results" / VERSION / "tables"
ROBUSTNESS_DIR = ROOT / "results" / VERSION / "robustness"
PAPER_TABLE_DIR = ROOT / "paper" / "tables"
CAPACITIES = (1, 2, 3, 8)


def _write_table(frame: pd.DataFrame, path: Path, *, caption: str, label: str,
                 column_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = frame.to_latex(index=False, escape=False, column_format=column_format)
    path.write_text(
        "\\begin{table}\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{latex}\\end{{table}}\n",
        encoding="utf-8",
    )


def table3_frame() -> pd.DataFrame:
    """Return capacity-specific absolute-total peaks with 48-week percentages."""
    grid = pd.read_csv(ROBUSTNESS_DIR / "same_horizon_normalization_full_grid.csv")
    # This report is deliberately main-only, so its CSV carries no
    # specification column.
    main = grid.copy()
    rows: list[dict[str, float | int]] = []
    for capacity in CAPACITIES:
        group = main.loc[main["capacity"].eq(capacity)]
        peak = group.loc[group["delta_total"].idxmax()]
        if not pd.notna(peak["delta_total_pct_myopic_same_horizon_48w"]):
            raise AssertionError(f"B={capacity}: missing 48-week normalized contrast.")
        rows.append({
            "B": capacity,
            r"Peak $\\lambda$": float(peak["reimbursement_share"]),
            r"$\\Delta^{total}$": float(peak["delta_total"]),
            r"$\\Delta^{total}$ (% of 48-week myopic profit)": float(
                peak["delta_total_pct_myopic_same_horizon_48w"]
            ),
        })
    return pd.DataFrame(rows)


def table4_frame() -> pd.DataFrame:
    """Return the existing compact full-reoptimization robustness comparison."""
    return pd.read_csv(TABLE_DIR / "robustness_peak_location_and_magnitude.csv")


def table5_frame() -> pd.DataFrame:
    """Return fixed-calendar behavioral-parameter uncertainty at main peaks."""
    return pd.read_csv(
        ROBUSTNESS_DIR / "fixed_calendar_behavioral_uncertainty_peak_summary.csv"
    )


def build_paper_tables() -> dict[str, pd.DataFrame]:
    """Write Tables 3--5 and return exactly the frames used for each export."""
    table3, table4, table5 = table3_frame(), table4_frame(), table5_frame()

    table3_tex = table3.copy()
    table3_tex[r"Peak $\\lambda$"] = table3_tex[r"Peak $\\lambda$"].map("{:.2f}".format)
    table3_tex[r"$\\Delta^{total}$"] = table3_tex[r"$\\Delta^{total}$"].map("{:.1f}".format)
    table3_tex[r"$\\Delta^{total}$ (% of 48-week myopic profit)"] = (
        table3_tex[r"$\\Delta^{total}$ (% of 48-week myopic profit)"].map("{:.2f}\\%".format)
    )
    _write_table(
        table3_tex, PAPER_TABLE_DIR / "table3_main_policy_peaks.tex",
        caption="Capacity-specific absolute total-contrast maxima.",
        label="tab:main_policy_peaks", column_format="rrrr",
    )
    _write_table(
        table4, PAPER_TABLE_DIR / "table4_robustness_peaks.tex",
        caption=("Robustness of peak total policy contrasts. Each cell reports peak "
                 "reimbursement share / peak contrast as a percentage of the "
                 "specification-specific 12-week myopic planning profit."),
        label="tab:robustness_peaks", column_format="lrrrr",
    )
    table5_tex = table5.copy()
    for column in ("p05", "p50", "p95"):
        table5_tex[column] = table5_tex[column].map("{:.1f}".format)
    _write_table(
        table5_tex, PAPER_TABLE_DIR / "table5_behavioral_uncertainty.tex",
        caption=("Behavioral-parameter uncertainty conditional on the selected "
                 "calendars; values are weighted 5th, 50th, and 95th percentiles."),
        label="tab:behavioral_uncertainty", column_format="rrlrrrlr",
    )
    return {"table3": table3, "table4": table4, "table5": table5}


if __name__ == "__main__":
    frames = build_paper_tables()
    for name, frame in frames.items():
        print(f"Wrote {name}: {len(frame)} rows")
