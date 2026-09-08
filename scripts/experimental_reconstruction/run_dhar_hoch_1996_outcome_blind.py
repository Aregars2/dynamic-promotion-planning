"""Outcome-blind search for Dhar and Hoch (1996) Coupon versus Bonus Buy tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dynamic_promotion_planning.dhar_hoch_reconstruction import (
    DEFAULT_SEARCH_WEEKS, load_windowed_treatment_side,
    price_summary, store_assignment, upc_week_architecture,
    scan_upc_week_architecture,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "experimental_reconstruction" / "dhar_hoch_1996_v1"
CATEGORIES = {"analgesics": "wana_csv(1).zip", "beer": "wber.zip", "rte_cereal": "wcer.csv", "oral_care": "wtbr.zip", "soft_drinks": "wsdr.zip"}


def attach_cereal_metadata(table: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.read_csv(
        RAW / "upccer.csv", usecols=["UPC", "DESCRIP", "SIZE", "CASE", "NITEM"], encoding="latin-1"
    )
    return table.merge(metadata.rename(columns={"UPC": "upc"}), on="upc", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Include adjacent weeks strictly for the one-week diagnostic; never MOVE.
    rows_weeks = range(min(DEFAULT_SEARCH_WEEKS) - 1, max(DEFAULT_SEARCH_WEEKS) + 2)
    all_ranked: list[pd.DataFrame] = []
    cereal_rows: pd.DataFrame | None = None
    for category, filename in CATEGORIES.items():
        ranked = scan_upc_week_architecture(RAW / filename, rows_weeks, chunksize=args.chunk_rows)
        ranked = ranked.loc[ranked.week.isin(DEFAULT_SEARCH_WEEKS)].copy()
        ranked.insert(0, "category", category)
        if category == "rte_cereal":
            ranked = attach_cereal_metadata(ranked)
            cereal_rows = load_windowed_treatment_side(RAW / filename, [177], chunksize=args.chunk_rows)
        ranked.to_csv(args.output_dir / f"{category}_upc_week_candidates.csv", index=False)
        all_ranked.append(ranked)
    pd.concat(all_ranked, ignore_index=True).sort_values("architecture_score", ascending=False).to_csv(
        args.output_dir / "all_category_upc_week_candidates.csv", index=False
    )
    all_candidates = pd.concat(all_ranked, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for category, group in all_candidates.groupby("category", sort=True):
        direct = group.loc[group.b_stores.ge(35) & group.c_stores.ge(35)]
        summary_rows.append({
            "category": category,
            "candidate_upc_weeks": len(group),
            "direct_near_43_43_candidates": len(direct),
            "best_direct_architecture_score": direct.architecture_score.max() if not direct.empty else None,
            "upc_metadata_available": category == "rte_cereal",
            "reconstruction_status": "unresolved_no_direct_near_43_43_B_C_pattern",
        })
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "category_reconstruction_summary.csv", index=False)
    if cereal_rows is None:
        raise RuntimeError("Cereal source was not loaded.")
    cheerios = 1600066610
    assignment = store_assignment(cereal_rows, upcs=[cheerios], week=177)
    assignment.insert(0, "category", "rte_cereal")
    assignment.insert(1, "experiment_week", 177)
    assignment.insert(2, "target_upc", cheerios)
    assignment.insert(3, "target_brand", "CHEERIOS")
    assignment.insert(4, "package_size", "15 OZ")
    assignment["evidence_source"] = "published Figure 1 clue plus DFF treatment-side fields"
    assignment["notes"] = "high-priority candidate only; no B/C randomized pattern at this UPC-week means not reconstructed"
    assignment.to_csv(args.output_dir / "week177_cheerios15oz_store_assignments.csv", index=False)
    price_summary(cereal_rows, upcs=[cheerios], week=177).to_csv(args.output_dir / "week177_cheerios15oz_price_summary.csv", index=False)
    summary = assignment.assignment.value_counts(dropna=False).to_dict()
    direct_count = int(((all_candidates.b_stores >= 35) & (all_candidates.c_stores >= 35)).sum())
    (args.output_dir / "RECONSTRUCTION_STATUS.md").write_text(
        "# Dhar and Hoch (1996) outcome-blind reconstruction\n\n"
        "Status: **not frozen**. All rankings use only UPC/store/week/quantity/price/SALE/OK fields and UPC metadata where available. "
        "No MOVE or other outcome was read.\n\n"
        f"The high-priority Cheerios 15 oz, week 177 check yielded assignment counts {summary}. "
        "It is therefore retained as an investigated clue, not a recovered Coupon-versus-Bonus-Buy experiment. "
        f"Across the initial search window, {direct_count} UPC-week observations have even a near 43/43 B/C split within the same UPC; "
        "therefore no category is frozen. The four other category candidate lists are UPC-only until their official UPC metadata files are supplied.\n",
        encoding="utf-8",
    )
    print(f"Wrote outcome-blind Dhar-Hoch candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
