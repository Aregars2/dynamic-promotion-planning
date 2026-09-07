"""Write enhanced outcome-blind diagnostics without changing candidate artifacts.

Run after the existing Phase 0 and candidate reconstruction stages. Outputs are
written to a separate diagnostics directory and contain no demand outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dynamic_promotion_planning.experimental_reconstruction import (
    coverage_summary,
    cross_category_agreement,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE0 = ROOT / "results" / "experimental_reconstruction" / "phase0"
DEFAULT_CANDIDATES = ROOT / "results" / "experimental_reconstruction" / "reconstruction_v1"
DEFAULT_OUTPUT = ROOT / "results" / "experimental_reconstruction" / "diagnostics_v1"


def ambiguity_summary(study: str, windows: pd.DataFrame) -> pd.DataFrame:
    top = windows.head(10).copy()
    if top.empty:
        return pd.DataFrame(columns=["study", "candidate_rank", "candidate_start_week", "candidate_end_week", "score", "score_gap_to_best", "ambiguity_status"])
    best = float(top.score.iloc[0])
    output = top.loc[:, [column for column in ["candidate_rank", "candidate_start_week", "candidate_end_week", "score", "price_shift_spread"] if column in top]].copy()
    output.insert(0, "study", study)
    output["score_gap_to_best"] = best - output.score
    output["ambiguity_status"] = output.score_gap_to_best.abs().le(1e-9).map({True: "numerical_top_tie", False: "ranked_price_side_candidate"})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-dir", type=Path, default=DEFAULT_PHASE0)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coverage = pd.read_csv(args.phase0_dir / "store_category_coverage.csv")
    coverage_summary(coverage).to_csv(args.output_dir / "price_side_coverage_summary.csv", index=False)
    study1 = pd.read_csv(args.candidate_dir / "study1_window.csv")
    study2 = pd.read_csv(args.candidate_dir / "study2_window.csv")
    pd.concat([ambiguity_summary("Study 1", study1), ambiguity_summary("Study 2", study2)], ignore_index=True).to_csv(
        args.output_dir / "candidate_ambiguity_summary.csv", index=False
    )
    evidence = pd.read_csv(args.candidate_dir / "study2_store_category_evidence.csv")
    cross_category_agreement(evidence).to_csv(args.output_dir / "study2_cross_category_agreement.csv", index=False)
    loo = pd.read_csv(args.candidate_dir / "study2_assignment_robustness.csv")
    pd.DataFrame(
        [{
            "evidence_universe_categories": int(len(loo)),
            "leave_one_out_start_min": int(loo.price_only_candidate_start_week.min()),
            "leave_one_out_start_max": int(loo.price_only_candidate_start_week.max()),
            "leave_one_out_unique_starts": int(loo.price_only_candidate_start_week.nunique()),
            "interpretation": "price_side_stability_diagnostic_not_historical_frame_or_assignment",
        }]
    ).to_csv(args.output_dir / "study2_leave_one_category_out_summary.csv", index=False)
    print(f"Wrote outcome-blind diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
