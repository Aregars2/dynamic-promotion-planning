"""Outcome-blind infrastructure for reconstructing historical price experiments.

This module deliberately separates externally documented historical constraints
from price-side evidence.  It contains no outcome reader.  Causal analysis is
available only through :func:`require_passed_freeze`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


RECONSTRUCTION_FIELDS = ("STORE", "UPC", "WEEK", "PRICE", "SALE", "OK")
OUTCOME_FIELDS = frozenset({"MOVE", "QTY", "PROFIT", "PROFIT_HEX", "SALES", "REVENUE"})
WEEK_ONE_START = date(1989, 9, 14)
EVIDENCE_COLUMNS = (
    "evidence_id", "study", "evidence_type", "source", "locator",
    "documented_start_date", "documented_end_date", "unit_type", "unit_id",
    "frame_status", "assignment", "confidence", "independent_of_outcomes", "count_implied",
)


class OutcomeFirewallError(ValueError):
    """Raised when pre-freeze code is offered an outcome column or file."""


class FreezeGateError(RuntimeError):
    """Raised when causal code is requested before a valid reconstruction freeze."""


def assert_outcome_blind(columns: Iterable[str]) -> tuple[str, ...]:
    """Validate and return reconstruction-safe columns, rejecting outcomes first."""
    actual = tuple(columns)
    forbidden = sorted(OUTCOME_FIELDS.intersection(actual))
    if forbidden:
        raise OutcomeFirewallError(f"Outcome fields are prohibited before freeze: {forbidden}")
    unknown = sorted(set(actual).difference(RECONSTRUCTION_FIELDS))
    if unknown:
        raise OutcomeFirewallError(f"Pre-freeze reader received non-permitted fields: {unknown}")
    return actual


def read_outcome_blind_csv(path: Path, *, usecols: Iterable[str] = RECONSTRUCTION_FIELDS, **kwargs: object) -> pd.DataFrame:
    """Read a scanner file only through an explicit safe allow-list."""
    safe = assert_outcome_blind(usecols)
    return pd.read_csv(path, usecols=list(safe), **kwargs)


@dataclass(frozen=True)
class WeekDecoder:
    """Official Dominick's week decoder (week 1 starts 1989-09-14)."""

    week_one_start: date = WEEK_ONE_START

    def week_bounds(self, week: int) -> tuple[date, date]:
        if int(week) < 1:
            raise ValueError("Dominick's week numbers must be positive.")
        start = self.week_one_start + timedelta(days=7 * (int(week) - 1))
        return start, start + timedelta(days=6)

    def date_to_week(self, value: date) -> int:
        if value < self.week_one_start:
            raise ValueError("Date precedes the official Dominick's decoder.")
        return (value - self.week_one_start).days // 7 + 1


@dataclass(frozen=True)
class HistoricalConstraints:
    """Externally documented constraints; empty values deliberately stay unresolved."""

    evidence: pd.DataFrame
    decoder: WeekDecoder = WeekDecoder()

    def windows(self, study: str) -> pd.DataFrame:
        rows = self.evidence.loc[
            self.evidence.study.eq(study)
            & self.evidence.evidence_type.eq("calendar_window")
            & self.evidence.independent_of_outcomes
        ].copy()
        if rows.empty:
            return rows
        rows["start_week"] = rows.documented_start_date.map(self.decoder.date_to_week)
        rows["end_week"] = rows.documented_end_date.map(self.decoder.date_to_week)
        return rows

    def admissible_starts(self, study: str, window_weeks: int) -> set[int] | None:
        windows = self.windows(study)
        if windows.empty:
            return None
        starts: set[int] = set()
        for row in windows.itertuples(index=False):
            starts.update(range(int(row.start_week), int(row.end_week) - window_weeks + 2))
        return starts

    def membership(self, study: str, unit_type: str) -> pd.DataFrame:
        return self.evidence.loc[
            self.evidence.study.eq(study)
            & self.evidence.unit_type.eq(unit_type)
            & self.evidence.frame_status.notna()
        ].copy()


def read_historical_evidence(path: Path) -> HistoricalConstraints:
    """Read the structured historical-evidence CSV without accepting outcomes."""
    evidence = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = sorted(set(EVIDENCE_COLUMNS).difference(evidence.columns))
    if missing:
        raise ValueError(f"Historical evidence is missing required columns: {missing}")
    evidence = evidence.loc[:, list(EVIDENCE_COLUMNS)].copy()
    for column in ("documented_start_date", "documented_end_date"):
        evidence[column] = pd.to_datetime(evidence[column].replace("", pd.NA), errors="raise").dt.date
    for column in ("independent_of_outcomes", "count_implied"):
        values = evidence[column].str.lower()
        if not values.isin({"true", "false"}).all():
            raise ValueError(f"{column} must contain only true or false.")
        evidence[column] = values.eq("true")
    outcome_words = evidence.astype("string").apply(lambda column: column.str.contains("move|profit|outcome|revenue", case=False, na=False)).any(axis=1)
    if outcome_words.any():
        bad = evidence.loc[outcome_words, "evidence_id"].tolist()
        raise OutcomeFirewallError(f"Historical evidence cannot use outcome material: {bad}")
    date_rows = evidence.evidence_type.eq("calendar_window")
    if evidence.loc[date_rows, ["documented_start_date", "documented_end_date"]].isna().any(axis=None):
        raise ValueError("calendar_window evidence requires start and end dates.")
    return HistoricalConstraints(evidence=evidence)


def candidate_diagnostics(candidates: pd.DataFrame, *, study: str, constraints: HistoricalConstraints | None, window_weeks: int) -> pd.DataFrame:
    """Attach deterministic admissibility and ambiguity diagnostics to candidates."""
    output = candidates.copy()
    allowed = None if constraints is None else constraints.admissible_starts(study, window_weeks)
    output["historical_constraint_supplied"] = allowed is not None
    output["historically_admissible"] = pd.NA if allowed is None else output.candidate_start_week.isin(allowed)
    if "score" in output:
        output["score_gap_to_next"] = output.score - output.score.shift(-1)
        output["near_top_score"] = output.score_gap_to_next.abs().le(1e-9)
    return output


def coverage_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    """Summarize price-side availability by category without demand outcomes."""
    required = {"category_code", "store", "first_week", "last_week", "valid_ok_rows", "observed_price_rows"}
    missing = required.difference(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table missing: {sorted(missing)}")
    result = coverage.groupby("category_code", as_index=False).agg(
        stores=("store", "nunique"),
        first_week=("first_week", "min"), last_week=("last_week", "max"),
        valid_price_rows=("observed_price_rows", "sum"), valid_ok_rows=("valid_ok_rows", "sum"),
    )
    result["price_row_share_of_valid"] = result.valid_price_rows / result.valid_ok_rows.where(result.valid_ok_rows.ne(0))
    return result


def cross_category_agreement(evidence: pd.DataFrame) -> pd.DataFrame:
    """Report signed price-shift agreement at each store, never infer treatment."""
    required = {"store", "category_code", "log_price_shift"}
    missing = required.difference(evidence.columns)
    if missing:
        raise ValueError(f"Category evidence missing: {sorted(missing)}")
    copy = evidence.loc[:, ["store", "category_code", "log_price_shift"]].dropna().copy()
    copy["sign"] = copy.log_price_shift.gt(0).astype(int) - copy.log_price_shift.lt(0).astype(int)
    summary = copy.groupby("store", as_index=False).agg(
        categories=("category_code", "nunique"),
        median_log_price_shift=("log_price_shift", "median"),
        positive_categories=("sign", lambda values: int((values > 0).sum())),
        negative_categories=("sign", lambda values: int((values < 0).sum())),
    )
    summary["sign_agreement_share"] = summary[["positive_categories", "negative_categories"]].max(axis=1) / summary.categories
    summary["interpretation"] = "price_side_agreement_not_treatment_assignment"
    return summary


def reconstruct_hyper_assignment(
    study2_store_labels: pd.DataFrame, hyper_evidence: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Apply externally documented Hyper labels without count completion.

    A missing, incomplete, or non-independent Hyper record leaves every eligible
    store unresolved.  Individual UPC promotion records are never inputs here.
    """
    required = {"store", "frame_status", "assignment"}
    missing = required.difference(study2_store_labels.columns)
    if missing:
        raise ValueError(f"Study 2 labels missing: {sorted(missing)}")
    eligible = study2_store_labels.loc[
        study2_store_labels.frame_status.eq("in_frame")
        & study2_store_labels.assignment.isin(["Control", "Hi-Lo"]), ["store", "assignment"]
    ].copy()
    eligible["hyper_assignment"] = "unresolved"
    eligible["reconstruction_status"] = "unresolved_no_independent_hyper_assignment"
    if hyper_evidence is None or hyper_evidence.empty:
        return eligible
    needed = {"unit_id", "assignment", "independent_of_outcomes", "count_implied"}
    missing = needed.difference(hyper_evidence.columns)
    if missing:
        raise ValueError(f"Hyper evidence missing: {sorted(missing)}")
    supported = hyper_evidence.loc[
        hyper_evidence.independent_of_outcomes & ~hyper_evidence.count_implied,
        ["unit_id", "assignment"],
    ].drop_duplicates("unit_id")
    mapping = dict(zip(pd.to_numeric(supported.unit_id, errors="coerce"), supported.assignment))
    eligible["hyper_assignment"] = eligible.store.map(mapping).fillna("unresolved")
    eligible["reconstruction_status"] = eligible.hyper_assignment.map(
        lambda value: "independently_documented" if value != "unresolved" else "unresolved_no_independent_hyper_assignment"
    )
    return eligible


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_freeze_manifest(*, output_dir: Path, files: Iterable[Path], protocol_version: str, evidence: pd.DataFrame, expected_categories: int = 26, expected_stores: int = 86) -> dict[str, object]:
    """Build a Phase 4 manifest; pass only with independently resolved frame labels."""
    category = evidence.loc[(evidence.unit_type == "category") & evidence.frame_status.notna()]
    store = evidence.loc[(evidence.unit_type == "store") & evidence.frame_status.notna()]
    independently_resolved = evidence.independent_of_outcomes.all() and not evidence.count_implied.any()
    passed = (
        independently_resolved
        and int((category.frame_status == "in_frame").sum()) == expected_categories
        and int((store.frame_status == "in_frame").sum()) == expected_stores
        and not category.frame_status.eq("unresolved").any()
        and not store.frame_status.eq("unresolved").any()
    )
    manifest = {
        "phase": 4,
        "protocol_version": protocol_version,
        "freeze_status": "passed" if passed else "blocked",
        "outcome_fields_loaded": False,
        "count_completion_used": bool(evidence.count_implied.any()),
        "independently_resolved": bool(independently_resolved),
        "in_frame_categories": int((category.frame_status == "in_frame").sum()),
        "in_frame_stores": int((store.frame_status == "in_frame").sum()),
        "expected_categories": expected_categories,
        "expected_stores": expected_stores,
        "files": {path.name: sha256_file(path) for path in sorted(files)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase4_freeze_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def require_passed_freeze(manifest_path: Path) -> dict[str, object]:
    """Hard Phase 5 gate. Call this before any outcome reader is constructed."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("freeze_status") != "passed":
        raise FreezeGateError("Phase 5 is prohibited: reconstruction freeze_status is not passed.")
    if manifest.get("outcome_fields_loaded"):
        raise FreezeGateError("Phase 5 is prohibited: freeze provenance is contaminated by outcomes.")
    if manifest.get("count_completion_used"):
        raise FreezeGateError("Phase 5 is prohibited: frame depends on count-implied labels.")
    return manifest
