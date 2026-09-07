"""Run Phases 1--4 of the Dominick's reconstruction without outcomes.

This script is deliberately conservative. It creates price-side candidate
windows and evidence tables, but it cannot freeze a treatment assignment until
published calendar timing and the historical Study 2 frame are independently
resolved. It never reads MOVE, QTY, PROFIT, or PROFIT_HEX.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "results" / "experimental_reconstruction" / "reconstruction_v1"
BASE_COMMIT = "7d98716144190a4e6b67459e299081fc397d05bb"
ALLOWED_FIELDS = ("STORE", "UPC", "WEEK", "PRICE", "SALE", "OK")
FORBIDDEN_FIELDS = {"MOVE", "QTY", "PROFIT", "PROFIT_HEX"}
WINDOW_WEEKS = 16
MIN_WINDOW_WEEKS = 12
EVIDENCE_THRESHOLD_LOG_PRICE = 0.025


def movement_member(archive: Path) -> str:
    from zipfile import ZipFile

    with ZipFile(archive) as handle:
        members = [name for name in handle.namelist() if name.lower().endswith(".csv")]
    if len(members) != 1:
        raise ValueError(f"Expected exactly one CSV in {archive.name}: {members!r}")
    return members[0]


def category_code(archive: Path) -> str:
    stem = Path(movement_member(archive)).stem.lower()
    return stem[1:] if stem.startswith("w") else stem


def checked_reader(archive: Path, chunksize: int) -> pd.io.parsers.TextFileReader:
    """Return an explicit allow-list reader; outcomes cannot enter memory."""
    from zipfile import ZipFile

    member = movement_member(archive)
    with ZipFile(archive) as handle:
        with handle.open(member) as raw:
            fields = set(raw.readline().decode("latin-1").strip().split(","))
    if not FORBIDDEN_FIELDS.issubset(fields):
        raise ValueError(f"Unexpected raw schema in {archive.name}: {sorted(fields)!r}")
    if not set(ALLOWED_FIELDS).issubset(fields):
        raise ValueError(f"Missing reconstruction fields in {archive.name}: {sorted(fields)!r}")
    return pd.read_csv(
        archive,
        compression="zip",
        usecols=list(ALLOWED_FIELDS),
        chunksize=chunksize,
        low_memory=False,
    )


def price_indices(archive: Path, chunksize: int, collect_bonus_buy: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return store-week log-price indices and optional price-side Bonus Buy candidates."""
    weekly: list[pd.DataFrame] = []
    episodes: list[pd.DataFrame] = []
    for chunk in checked_reader(archive, chunksize):
        price = pd.to_numeric(chunk["PRICE"], errors="coerce")
        valid = pd.to_numeric(chunk["OK"], errors="coerce").eq(1)
        usable = chunk.loc[valid & price.gt(0), ["STORE", "UPC", "WEEK", "SALE"]].copy()
        usable["log_price"] = np.log(price.loc[usable.index])
        usable["marked_sale"] = usable["SALE"].notna()
        usable["regular_log_price"] = usable["log_price"].where(~usable["marked_sale"], 0.0)
        usable["regular_price_row"] = (~usable["marked_sale"]).astype(int)
        grouped = (
            usable.groupby(["STORE", "WEEK"], as_index=False)
            .agg(
                log_price_sum=("log_price", "sum"),
                price_rows=("log_price", "size"),
                regular_log_price_sum=("regular_log_price", "sum"),
                regular_price_rows=("regular_price_row", "sum"),
                marked_sale_rows=("marked_sale", "sum"),
            )
        )
        weekly.append(grouped)
        if collect_bonus_buy:
            bonus = usable.loc[usable["SALE"].eq("B"), ["STORE", "UPC", "WEEK", "log_price"]]
            if not bonus.empty:
                episodes.append(
                    bonus.groupby(["UPC", "WEEK"], as_index=False)
                    .agg(marked_bonus_buy_store_count=("STORE", "nunique"), marked_bonus_buy_rows=("STORE", "size"), mean_log_price=("log_price", "mean"))
                )
    combined = pd.concat(weekly, ignore_index=True)
    combined = combined.groupby(["STORE", "WEEK"], as_index=False).sum(numeric_only=True)
    combined["mean_log_price"] = combined["log_price_sum"] / combined["price_rows"]
    combined["mean_regular_log_price"] = np.where(
        combined["regular_price_rows"].gt(0),
        combined["regular_log_price_sum"] / combined["regular_price_rows"],
        np.nan,
    )
    bonus_output = pd.DataFrame(
        columns=["UPC", "WEEK", "marked_bonus_buy_store_count", "marked_bonus_buy_rows", "mean_log_price"]
    )
    if episodes:
        bonus_output = pd.concat(episodes, ignore_index=True).groupby(["UPC", "WEEK"], as_index=False).agg(
            marked_bonus_buy_store_count=("marked_bonus_buy_store_count", "max"),
            marked_bonus_buy_rows=("marked_bonus_buy_rows", "sum"),
            mean_log_price=("mean_log_price", "mean"),
        )
    return combined, bonus_output


def window_deltas(index: pd.DataFrame, start: int, value: str) -> pd.Series:
    baseline = index.loc[index.WEEK.between(start - WINDOW_WEEKS, start - 1)]
    treatment = index.loc[index.WEEK.between(start, start + WINDOW_WEEKS - 1)]
    before = baseline.groupby("STORE")[value].agg(["mean", "count"])
    after = treatment.groupby("STORE")[value].agg(["mean", "count"])
    joined = before.join(after, how="inner", lsuffix="_before", rsuffix="_after")
    joined = joined.loc[(joined.count_before >= MIN_WINDOW_WEEKS) & (joined.count_after >= MIN_WINDOW_WEEKS)]
    return joined.mean_after - joined.mean_before


def rank_cereal_windows(index: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for start in range(WINDOW_WEEKS + 1, int(index.WEEK.max()) - WINDOW_WEEKS + 2):
        delta = window_deltas(index, start, "mean_regular_log_price").dropna()
        if len(delta) < 40:
            continue
        spread = float(np.quantile(delta, 0.9) - np.quantile(delta, 0.1))
        center = float(np.median(np.abs(delta - np.median(delta))))
        records.append({"candidate_start_week": start, "candidate_end_week": start + WINDOW_WEEKS - 1, "n_stores": len(delta), "price_shift_spread": spread, "score": spread / (center + 1e-6)})
    result = pd.DataFrame(records).sort_values(["score", "price_shift_spread"], ascending=False).reset_index(drop=True)
    result["candidate_rank"] = np.arange(1, len(result) + 1)
    result["selection_status"] = "price_only_candidate_not_frozen"
    return result


def price_evidence(delta: pd.Series) -> pd.DataFrame:
    center = float(delta.median())
    evidence = pd.DataFrame({"store": delta.index.astype(int), "log_price_shift": delta.values})
    relative = evidence.log_price_shift - center
    evidence["price_evidence"] = np.select(
        [relative <= -EVIDENCE_THRESHOLD_LOG_PRICE, relative >= EVIDENCE_THRESHOLD_LOG_PRICE],
        ["EDLP-supporting", "Hi-Lo-supporting"],
        default="Control-supporting",
    )
    evidence["evidence_strength"] = np.abs(relative)
    evidence["assignment"] = "unresolved"
    evidence["assignment_confidence"] = "not_frozen_price_side_evidence_only"
    return evidence


def markdown_first_stage(archive: Path, start: int, chunksize: int) -> pd.DataFrame:
    """Price-only markdown diagnostics at a candidate window; not an IV estimate."""
    rows: list[pd.DataFrame] = []
    for chunk in checked_reader(archive, chunksize):
        price = pd.to_numeric(chunk["PRICE"], errors="coerce")
        valid = pd.to_numeric(chunk["OK"], errors="coerce").eq(1)
        week = pd.to_numeric(chunk["WEEK"], errors="coerce")
        selected = chunk.loc[valid & price.gt(0) & week.between(start - WINDOW_WEEKS, start + WINDOW_WEEKS - 1), ["STORE", "UPC", "WEEK", "SALE"]].copy()
        selected["PRICE"] = price.loc[selected.index]
        selected["period"] = np.where(selected.WEEK < start, "reference", "candidate")
        selected["unmarked_sale"] = selected.SALE.isna()
        rows.append(selected)
    data = pd.concat(rows, ignore_index=True)
    reference = data.loc[(data.period == "reference") & data.unmarked_sale].groupby(["STORE", "UPC"], as_index=False).PRICE.median().rename(columns={"PRICE": "reference_price"})
    candidate = data.loc[data.period == "candidate"].merge(reference, on=["STORE", "UPC"], how="inner")
    candidate["markdown_depth"] = 1 - candidate.PRICE / candidate.reference_price
    candidate = candidate.loc[candidate.markdown_depth.between(-0.5, 0.95)]
    return candidate.groupby("STORE", as_index=False).agg(
        price_observations=("markdown_depth", "size"),
        mean_markdown_depth=("markdown_depth", "mean"),
        median_markdown_depth=("markdown_depth", "median"),
        share_positive_markdown=("markdown_depth", lambda values: float((values > 0).mean())),
    ).rename(columns={"STORE": "store"})


def precompute_category_deltas(indices: pd.DataFrame) -> dict[tuple[int, str], pd.Series]:
    """Cache price-only store shifts for every category/window combination."""
    cache: dict[tuple[int, str], pd.Series] = {}
    max_week = int(indices.WEEK.max())
    for category, group in indices.groupby("category_code", sort=False):
        for start in range(WINDOW_WEEKS + 1, max_week - WINDOW_WEEKS + 2):
            delta = window_deltas(group, start, "mean_regular_log_price").dropna()
            if not delta.empty:
                cache[(start, str(category))] = delta
    return cache


def study2_candidate_windows(
    cache: dict[tuple[int, str], pd.Series], categories: list[str], min_categories: int = 12
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Rank cached price-side windows without selecting a final Study 2 window."""
    records: list[dict[str, float | int]] = []
    category_store_deltas: dict[int, pd.DataFrame] = {}
    starts = sorted({start for start, category in cache if category in categories})
    for start in starts:
        parts: list[pd.DataFrame] = []
        for category in categories:
            delta = cache.get((start, category))
            if delta is not None and not delta.empty:
                parts.append(pd.DataFrame({"category_code": category, "store": delta.index, "log_price_shift": delta.values}))
        if not parts:
            continue
        details = pd.concat(parts, ignore_index=True)
        aggregate = details.groupby("store").agg(log_price_shift=("log_price_shift", "median"), usable_categories=("category_code", "nunique"))
        aggregate = aggregate.loc[aggregate.usable_categories >= min_categories]
        if len(aggregate) < 40:
            continue
        spread = float(np.quantile(aggregate.log_price_shift, 0.9) - np.quantile(aggregate.log_price_shift, 0.1))
        consensus = float(aggregate.log_price_shift.std(ddof=0))
        records.append({"candidate_start_week": start, "candidate_end_week": start + WINDOW_WEEKS - 1, "n_stores": len(aggregate), "median_usable_categories": float(aggregate.usable_categories.median()), "price_shift_spread": spread, "score": spread * consensus})
        category_store_deltas[start] = details
    if not records:
        raise ValueError(
            f"No price-side Study 2 candidates with at least {min_categories} usable categories."
        )
    ranked = pd.DataFrame(records).sort_values(["score", "price_shift_spread"], ascending=False).reset_index(drop=True)
    ranked["candidate_rank"] = np.arange(1, len(ranked) + 1)
    ranked["selection_status"] = "price_only_candidate_not_frozen"
    return ranked, category_store_deltas


def leave_one_category_out(cache: dict[tuple[int, str], pd.Series], all_categories: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for excluded in all_categories:
        ranked, _ = study2_candidate_windows(cache, [category for category in all_categories if category != excluded])
        best = ranked.iloc[0]
        rows.append({"excluded_category_code": excluded, "price_only_candidate_start_week": int(best.candidate_start_week), "price_only_candidate_end_week": int(best.candidate_end_week), "score": float(best.score), "selection_status": "not_frozen"})
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_freeze_blocker(output_dir: Path, outputs: list[Path]) -> None:
    manifest = pd.DataFrame(
        [
            {
                "freeze_status": "blocked",
                "base_protocol_commit": BASE_COMMIT,
                "blocking_condition": "Published calendar-week timing and independently reconstructed Study 2 26-category/86-store membership remain unresolved.",
                "outcome_fields_loaded": False,
                "count_completion_used": False,
                "file": path.name,
                "sha256": sha256(path),
            }
            for path in outputs
        ]
    )
    manifest.to_csv(output_dir / "reconstruction_manifest.csv", index=False)
    (output_dir / "reconstruction_hashes.txt").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in outputs) + "\n", encoding="utf-8"
    )
    (output_dir / "RECONSTRUCTION_PROTOCOL.md").write_text(
        "# Reconstruction status\n\n"
        "Status: **not frozen**. The files in this directory are price-side candidates only. "
        "They must not be used for causal analysis or policy calibration until calendar timing "
        "and the Study 2 experiment frame are independently resolved and a new manifest is frozen.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(args.raw_dir.glob("*.zip"))
    if len(archives) != 28:
        raise ValueError(f"Expected 28 public movement archives; found {len(archives)}")
    archive_by_code = {category_code(archive): archive for archive in archives}
    cereal_archive = archive_by_code.get("cer")
    if cereal_archive is None:
        raise ValueError("Cereal archive code 'cer' was not found.")

    cereal_index, bonus_buy = price_indices(cereal_archive, args.chunk_rows, collect_bonus_buy=True)
    cereal_windows = rank_cereal_windows(cereal_index)
    cereal_windows.to_csv(args.output_dir / "study1_window.csv", index=False)
    study1_start = int(cereal_windows.iloc[0].candidate_start_week)
    study1_delta = window_deltas(cereal_index, study1_start, "mean_regular_log_price").dropna()
    study1_assignments = price_evidence(study1_delta)
    study1_assignments["candidate_start_week"] = study1_start
    study1_assignments["frame_status"] = "unresolved"
    study1_assignments.to_csv(args.output_dir / "study1_store_assignments.csv", index=False)
    study1_assignments[["store", "log_price_shift", "price_evidence", "evidence_strength", "assignment"]].to_csv(args.output_dir / "study1_price_shifts.csv", index=False)
    first_stage = markdown_first_stage(cereal_archive, study1_start, args.chunk_rows).merge(study1_assignments[["store", "price_evidence", "assignment"]], on="store", how="left")
    first_stage["candidate_start_week"] = study1_start
    first_stage["interpretation"] = "price_only_markdown_diagnostic_not_IV"
    first_stage.to_csv(args.output_dir / "study1_first_stage_diagnostics.csv", index=False)

    all_indices: list[pd.DataFrame] = []
    for code, archive in archive_by_code.items():
        index, _ = price_indices(archive, args.chunk_rows)
        index["category_code"] = code
        all_indices.append(index[["category_code", "STORE", "WEEK", "mean_regular_log_price"]].rename(columns={"STORE": "store"}))
    study2_indices = pd.concat(all_indices, ignore_index=True).rename(columns={"store": "STORE"})
    all_categories = sorted(study2_indices.category_code.unique())
    delta_cache = precompute_category_deltas(study2_indices)
    study2_windows, category_deltas = study2_candidate_windows(delta_cache, all_categories)
    study2_windows.to_csv(args.output_dir / "study2_window.csv", index=False)
    study2_start = int(study2_windows.iloc[0].candidate_start_week)
    category_evidence = category_deltas[study2_start]
    aggregate_evidence = category_evidence.groupby("store").agg(log_price_shift=("log_price_shift", "median"), usable_categories=("category_code", "nunique"))
    study2_assignments = price_evidence(aggregate_evidence.log_price_shift)
    study2_assignments["usable_categories"] = study2_assignments.store.map(aggregate_evidence.usable_categories)
    study2_assignments["frame_status"] = "unresolved"
    study2_assignments["candidate_start_week"] = study2_start
    study2_assignments["count_implied"] = False
    study2_assignments.to_csv(args.output_dir / "study2_store_assignments.csv", index=False)
    category_evidence = category_evidence.merge(study2_assignments[["store", "price_evidence"]], on="store", how="left")
    category_evidence["frame_status"] = "unresolved"
    category_evidence.to_csv(args.output_dir / "study2_store_category_evidence.csv", index=False)
    category_evidence.groupby("category_code", as_index=False).agg(n_stores=("store", "nunique"), median_log_price_shift=("log_price_shift", "median"), price_shift_sd=("log_price_shift", "std")).assign(frame_status="unresolved").to_csv(args.output_dir / "study2_category_consensus.csv", index=False)
    leave_one_category_out(delta_cache, all_categories).to_csv(args.output_dir / "study2_assignment_robustness.csv", index=False)

    hyper = study2_assignments[["store", "frame_status"]].copy()
    hyper["hyper_assignment"] = "unresolved"
    hyper["randomization_status"] = "not_reconstructed"
    hyper["reason"] = "Study 2 frame and calendar window are not frozen"
    hyper.to_csv(args.output_dir / "rte_hyper_assignments.csv", index=False)
    bonus_buy["episode_type"] = "candidate_manager_selected_bonus_buy"
    bonus_buy["randomized_status"] = "not_randomized"
    bonus_buy["window_status"] = "unresolved"
    bonus_buy.to_csv(args.output_dir / "rte_hyper_candidate_episodes.csv", index=False)
    pd.DataFrame([{"candidate_episode_rows": len(bonus_buy), "identification_rule": "Observed SALE == B in a valid price row; absence of SALE is not evidence of no promotion.", "randomized_status": "not_randomized", "window_status": "unresolved"}]).to_csv(args.output_dir / "rte_hyper_episode_diagnostics.csv", index=False)

    (args.output_dir / "study1_reconstruction_audit.md").write_text("# Study 1 reconstruction audit\n\nPrice-side candidate windows and evidence labels were generated without outcomes. The experimental window and final EDLP/Control/Hi-Lo labels remain unresolved pending published calendar timing.\n", encoding="utf-8")
    (args.output_dir / "study2_reconstruction_audit.md").write_text("# Study 2 reconstruction audit\n\nAll 28 archives were used as price-side evidence. No unit was forced into the historical 26-category/86-store frame, and no 29/29/28 label completion was used.\n", encoding="utf-8")
    (args.output_dir / "rte_hyper_reconstruction_audit.md").write_text("# RTE Hyper reconstruction audit\n\nCandidate Bonus Buy episodes are manager-selected promotion candidates, not randomized treatments. Hyper assignment remains unresolved until the Study 2 frame is frozen.\n", encoding="utf-8")

    outputs = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name not in {"reconstruction_manifest.csv", "reconstruction_hashes.txt", "RECONSTRUCTION_PROTOCOL.md"})
    write_freeze_blocker(args.output_dir, outputs)
    print(f"Wrote price-side reconstruction candidates to {args.output_dir}")
    print("Freeze status: blocked; no outcome field was loaded and no count completion was used.")


if __name__ == "__main__":
    main()
