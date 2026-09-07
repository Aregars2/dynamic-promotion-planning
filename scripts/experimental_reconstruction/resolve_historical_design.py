"""Phase 0B: resolve published historical-design anchors without outcomes.

This inexpensive gate translates the official Dominick's week decoder and
records only independently documented design facts.  It deliberately does not
turn a price-side candidate into a treatment window, frame, or assignment.

Run:
    python scripts/experimental_reconstruction/resolve_historical_design.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_DIR = ROOT / "results" / "experimental_reconstruction" / "reconstruction_v1"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "experimental_reconstruction" / "phase0b_historical_design"
WEEK_ONE_START = date(1989, 9, 14)
PAPER_PUBLICATION_DATE = date(1994, 10, 1)


def decode_week(week: int) -> tuple[date, date]:
    """Use the official decoder's Week 1 anchor and seven-day weekly cadence."""
    start = WEEK_ONE_START + timedelta(days=7 * (int(week) - 1))
    return start, start + timedelta(days=6)


def candidate_dates(candidate_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for study, filename in (("Study 1", "study1_window.csv"), ("Study 2", "study2_window.csv")):
        path = candidate_dir / filename
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        if table.empty:
            continue
        candidate = table.iloc[0]
        start_week = int(candidate["candidate_start_week"])
        end_week = int(candidate["candidate_end_week"])
        start_date, _ = decode_week(start_week)
        _, end_date = decode_week(end_week)
        chronology = "not_historically_frozen"
        if study == "Study 1" and start_date > PAPER_PUBLICATION_DATE:
            chronology = "rejected_post_publication_false_positive_candidate"
        rows.append(
            {
                "study": study,
                "candidate_start_week": start_week,
                "candidate_end_week": end_week,
                "candidate_start_date": start_date.isoformat(),
                "candidate_end_date": end_date.isoformat(),
                "price_side_status": str(candidate.get("selection_status", "unknown")),
                "chronology_status": chronology,
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, candidates: pd.DataFrame) -> None:
    study1 = candidates.loc[candidates.study.eq("Study 1")]
    study2 = candidates.loc[candidates.study.eq("Study 2")]
    study1_text = "No Study 1 price-side candidate was available."
    if not study1.empty:
        row = study1.iloc[0]
        study1_text = (
            f"The retained unconstrained Study 1 candidate is weeks "
            f"{row.candidate_start_week}-{row.candidate_end_week} "
            f"({row.candidate_start_date} to {row.candidate_end_date}); its status is "
            f"`{row.chronology_status}`. It is not a treatment window."
        )
    study2_text = "No Study 2 price-side candidate was available."
    if not study2.empty:
        row = study2.iloc[0]
        study2_text = (
            f"The retained all-28-archive Study 2 candidate is weeks "
            f"{row.candidate_start_week}-{row.candidate_end_week} "
            f"({row.candidate_start_date} to {row.candidate_end_date}); its status remains "
            f"`{row.chronology_status}`."
        )
    (output_dir / "historical_design_resolution.md").write_text(
        "# Phase 0B historical design resolution\n\n"
        "## What is independently established\n\n"
        "The official Dominick's Data Manual decodes database week 1 as "
        "1989-09-14 to 1989-09-20. Hoch, Dreze, and Purk (1994) establish "
        "the Study 1 three-arm category-level design, Study 2's 26-category "
        "86-store design and reported 29/29/28 totals, and Study 3's separate "
        "category-level Hyper randomization. The paper gives Study 2 only as "
        "approximately eight months after Study 1; it does not supply calendar "
        "dates, a 26-category roster, or an 86-store roster.\n\n"
        "## Candidate chronology\n\n"
        f"{study1_text}\n\n{study2_text}\n\n"
        "## Decision\n\n"
        "Historical timing is not resolved to an admissible database-week interval, "
        "and the 26-category and 86-store Study 2 membership tables are unresolved. "
        "Therefore no constrained Phase 1 rerun, Study 2 label reconstruction, or "
        "Study 3 Hyper reconstruction is authorized. The prior price-side files are "
        "preserved as diagnostics only. No outcome field was read by this stage.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidate_dates(args.candidate_dir)
    candidates.to_csv(args.output_dir / "candidate_calendar_crosswalk.csv", index=False)
    frame = pd.DataFrame(
        [
            {"unit_type": "category", "unit": "all 28 public archives", "status": "evidence_universe_not_frame", "basis": "public archives"},
            {"unit_type": "category", "unit": "historical Study 2 26-category roster", "status": "unresolved", "basis": "no independently documented roster"},
            {"unit_type": "store", "unit": "historical Study 2 86-store roster", "status": "unresolved", "basis": "no independently documented roster"},
        ]
    )
    frame.to_csv(args.output_dir / "study2_frame_resolution.csv", index=False)
    status = {
        "phase": "0B_historical_design_resolution",
        "freeze_status": "blocked",
        "outcome_fields_loaded": False,
        "official_week_decoder": "Dominick's Data Manual, Part 8",
        "week_1_start": WEEK_ONE_START.isoformat(),
        "historical_timing_resolved": False,
        "study2_category_frame_resolved": False,
        "study2_store_frame_resolved": False,
        "count_completion_used": False,
        "next_authorized_action": "obtain independently dated design documentation and 26-category/86-store rosters",
    }
    (args.output_dir / "phase0b_manifest.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, candidates)
    print(f"Wrote Phase 0B historical-resolution gate to {args.output_dir}")
    print("Freeze status: blocked; no outcome field was loaded.")


if __name__ == "__main__":
    main()
