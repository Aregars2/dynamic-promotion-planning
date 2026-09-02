"""Fail-fast verification of the canonical paper-output contract."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_paper_tables import (  # noqa: E402
    PAPER_TABLE_DIR, ROBUSTNESS_DIR, TABLE_DIR, build_paper_tables,
)


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required paper output is missing: {path}")


def _close(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    # Verification targets are paper-rounded, while source outputs retain full
    # numerical precision.
    if not np.allclose(actual, expected, rtol=0.0, atol=5e-2):
        raise AssertionError(message)


def main() -> None:
    table1 = ROOT / "results" / "empirical_bayes" / "tables" / "paper_product_characteristics.csv"
    table1_tex = table1.with_suffix(".tex")
    table2 = ROOT / "results" / "final" / "tables" / "final_demand_model_comparison.csv"
    figure1 = ROOT / "results" / "empirical_bayes" / "figures" / "paper_promotion_depths_and_model_implied_post_effects.png"
    figure2 = ROOT / "results" / "empirical_bayes_price_consistent" / "figures" / "three_policy_components_by_capacity_lambda_050_to_100_48_week_normalized.png"
    table2_tex = ROOT / "paper" / "tables" / "table2_demand_model_performance.tex"
    for path in (table1, table1_tex, table2, table2_tex, figure1, figure2):
        _require(path)

    t1 = pd.read_csv(table1)
    t2 = pd.read_csv(table2)
    if len(t1) != 8 or t1.empty:
        raise AssertionError("Table 1 is not the eight-product canonical output.")
    if not {"model", "split", "mean_poisson_deviance"}.issubset(t2.columns):
        raise AssertionError("Table 2 does not have the canonical predictive-performance schema.")

    frames = build_paper_tables()
    for filename in ("table3_main_policy_peaks.tex", "table4_robustness_peaks.tex", "table5_behavioral_uncertainty.tex"):
        _require(PAPER_TABLE_DIR / filename)

    table3 = frames["table3"]
    _close(table3["B"].to_numpy(), np.array([1, 2, 3, 8]), "Table 3 capacity rows changed.")
    _close(table3[r"Peak $\\lambda$"].to_numpy(), np.array([0.81, 0.76, 0.86, 0.76]), "Figure-2/Table-3 peak lambdas changed.")
    _close(table3[r"$\\Delta^{total}$ (\\$)"].to_numpy(), np.array([1930.6, 2351.6, 2342.9, 2502.8]), "Table 3 dollar peaks changed.")
    _close(table3[r"$\\Delta^{total}$ (% of 48-week myopic profit)"].to_numpy(), np.array([0.33, 0.40, 0.40, 0.43]), "Table 3 48-week percentages changed.")

    cooldown = pd.read_csv(ROBUSTNESS_DIR / "cooldown_3_weeks" / "cooldown_3_weeks_full_grid.csv")
    peaks = cooldown.loc[cooldown.groupby("capacity")["delta_total"].idxmax()]
    if not peaks["reimbursement_share"].between(0.76, 0.79).all():
        raise AssertionError("Cooldown-3 peak locations no longer fall in the expected range.")
    pct48 = 100.0 * peaks["delta_total"] / peaks["myopic_profit"]
    if not pct48.between(0.25, 0.30).all():
        raise AssertionError("Cooldown-3 48-week peak magnitudes no longer fall in the expected range.")
    attenuation = 1.0 - pct48.to_numpy() / table3[r"$\\Delta^{total}$ (% of 48-week myopic profit)"].to_numpy()
    # B=1 is 23.3% before paper rounding; the remaining capacities are
    # 30.6--31.4%. This is the stated approximately 24--32% range.
    if not np.all((attenuation >= 0.23) & (attenuation <= 0.32)):
        raise AssertionError("Cooldown-3 attenuation no longer falls in the expected range.")
    print("Paper-output verification passed.")


if __name__ == "__main__":
    main()
