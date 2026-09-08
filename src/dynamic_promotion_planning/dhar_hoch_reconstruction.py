"""Outcome-blind candidate search for Dhar and Hoch (1996) experiments.

The allowed scanner columns intentionally exclude MOVE, PROFIT, and all
derived outcomes.  Candidate rankings use only the published one-week 43/43
Coupon/Bonus-Buy architecture, treatment-side codes, prices, quantities, and
UPC metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ALLOWED_MOVEMENT_FIELDS = ("STORE", "UPC", "WEEK", "QTY", "PRICE", "SALE", "OK")
FORBIDDEN_OUTCOME_FIELDS = frozenset({"MOVE", "PROFIT", "PROFIT_HEX", "PRICE_HEX"})
DEFAULT_SEARCH_WEEKS = range(151, 204)


class DharOutcomeFirewallError(ValueError):
    """Raised when this pre-freeze reconstruction is offered an outcome field."""


def assert_dhar_outcome_blind(columns: Iterable[str]) -> tuple[str, ...]:
    actual = tuple(columns)
    forbidden = sorted(FORBIDDEN_OUTCOME_FIELDS.intersection(actual))
    if forbidden:
        raise DharOutcomeFirewallError(f"Dhar-Hoch reconstruction cannot read outcome fields: {forbidden}")
    unexpected = sorted(set(actual).difference(ALLOWED_MOVEMENT_FIELDS))
    if unexpected:
        raise DharOutcomeFirewallError(f"Non-permitted pre-freeze fields: {unexpected}")
    return actual


def safe_movement_reader(path: Path, *, chunksize: int) -> pd.io.parsers.TextFileReader:
    """Return an allow-listed movement reader for CSV or one-CSV ZIP archives."""
    assert_dhar_outcome_blind(ALLOWED_MOVEMENT_FIELDS)
    kwargs: dict[str, object] = {"usecols": list(ALLOWED_MOVEMENT_FIELDS), "chunksize": chunksize, "low_memory": False}
    if path.suffix.lower() == ".zip":
        kwargs["compression"] = "zip"
    return pd.read_csv(path, **kwargs)


def load_windowed_treatment_side(path: Path, weeks: Iterable[int], *, chunksize: int) -> pd.DataFrame:
    """Load only rows in the candidate period through the outcome-blind reader."""
    wanted = {int(week) for week in weeks}
    pieces: list[pd.DataFrame] = []
    for chunk in safe_movement_reader(path, chunksize=chunksize):
        subset = chunk.loc[pd.to_numeric(chunk.WEEK, errors="coerce").isin(wanted)].copy()
        if not subset.empty:
            pieces.append(subset)
    if not pieces:
        return pd.DataFrame(columns=ALLOWED_MOVEMENT_FIELDS)
    return pd.concat(pieces, ignore_index=True)


def scan_upc_week_architecture(path: Path, weeks: Iterable[int], *, chunksize: int) -> pd.DataFrame:
    """Stream a candidate scan, retaining store sets rather than scanner rows.

    This is deliberately distinct from loading a large multiweek panel: all
    architecture statistics can be accumulated from treatment-side fields and
    bounded store sets, which keeps the real-data scan memory-safe.
    """
    wanted = {int(week) for week in weeks}
    # DFF store IDs are positive small integers. Bit masks retain exact store
    # membership while avoiding hundreds of thousands of Python set objects.
    stats: dict[tuple[int, int], dict[str, object]] = {}
    for chunk in safe_movement_reader(path, chunksize=chunksize):
        subset = chunk.loc[pd.to_numeric(chunk.WEEK, errors="coerce").isin(wanted)].copy()
        if subset.empty:
            continue
        subset = subset.loc[pd.to_numeric(subset.OK, errors="coerce").eq(1)]
        subset["sale_code"] = subset.SALE.fillna("").astype(str).str.strip().str.upper()
        for row in subset.itertuples(index=False):
            key = (int(row.UPC), int(row.WEEK))
            current = stats.setdefault(key, {"B": 0, "C": 0, "S": 0, "represented": 0, "missing": 0, "price_b": [], "price_c": []})
            store = int(row.STORE)
            store_bit = 1 << store
            current["represented"] |= store_bit  # type: ignore[index,operator]
            if row.sale_code in {"B", "C", "S"}:
                current[row.sale_code] |= store_bit  # type: ignore[index,operator]
            else:
                current["missing"] += 1  # type: ignore[index]
            if row.sale_code == "B" and pd.notna(row.PRICE):
                current["price_b"].append(float(row.PRICE))  # type: ignore[index]
            if row.sale_code == "C" and pd.notna(row.PRICE):
                current["price_c"].append(float(row.PRICE))  # type: ignore[index]
    records: list[dict[str, object]] = []
    for (upc, week), current in stats.items():
        b_mask, c_mask = int(current["B"]), int(current["C"])
        bc_mask = b_mask | c_mask
        b, c = b_mask.bit_count(), c_mask.bit_count()
        bc_count = bc_mask.bit_count()
        score = max(0.0, 1 - abs(b - 43) / 43) + max(0.0, 1 - abs(c - 43) / 43) + max(0.0, 1 - abs(bc_count - 86) / 86)
        records.append({
            "upc": upc, "week": week, "b_stores": b, "c_stores": c, "s_stores": int(current["S"]).bit_count(),
            "bc_stores": bc_count, "represented_stores": int(current["represented"]).bit_count(),
            "missing_sale_stores": int(current["missing"]), "bc_conflicting_stores": (b_mask & c_mask).bit_count(),
            "median_price_b": float(np.median(current["price_b"])) if current["price_b"] else np.nan,
            "median_price_c": float(np.median(current["price_c"])) if current["price_c"] else np.nan,
            "architecture_score": score, "rank_status": "outcome_blind_candidate_not_frozen",
        })
    output = pd.DataFrame(records)
    if output.empty:
        return output
    counts = {(int(row.upc), int(row.week)): int(row.bc_stores) for row in output.itertuples()}
    output["prior_week_bc_stores"] = [counts.get((int(row.upc), int(row.week) - 1), 0) for row in output.itertuples()]
    output["next_week_bc_stores"] = [counts.get((int(row.upc), int(row.week) + 1), 0) for row in output.itertuples()]
    output["one_week_price_side_support"] = output.prior_week_bc_stores.eq(0) & output.next_week_bc_stores.eq(0)
    return output.sort_values(["architecture_score", "bc_stores"], ascending=False).reset_index(drop=True)


def upc_week_architecture(rows: pd.DataFrame) -> pd.DataFrame:
    """Score every UPC-week by the published B/C randomization architecture."""
    data = rows.copy()
    data["valid"] = pd.to_numeric(data.OK, errors="coerce").eq(1)
    data = data.loc[data.valid]
    data["sale_code"] = data.SALE.fillna("").astype(str).str.strip().str.upper()
    records: list[dict[str, object]] = []
    for (upc, week), group in data.groupby(["UPC", "WEEK"], sort=True):
        stores_by_code = {code: set(group.loc[group.sale_code.eq(code), "STORE"].dropna().astype(int)) for code in ("B", "C", "S")}
        represented = set(group.STORE.dropna().astype(int))
        b, c = len(stores_by_code["B"]), len(stores_by_code["C"])
        bc_stores = stores_by_code["B"] | stores_by_code["C"]
        conflicts = stores_by_code["B"] & stores_by_code["C"]
        price_by_code = group.loc[group.sale_code.isin(["B", "C"])].groupby("sale_code").PRICE.median()
        architecture_score = max(0.0, 1 - abs(b - 43) / 43) + max(0.0, 1 - abs(c - 43) / 43) + max(0.0, 1 - abs(len(bc_stores) - 86) / 86)
        records.append({
            "upc": int(upc), "week": int(week), "b_stores": b, "c_stores": c,
            "s_stores": len(stores_by_code["S"]), "bc_stores": len(bc_stores),
            "represented_stores": len(represented), "missing_sale_stores": int((group.sale_code.eq("")).sum()),
            "bc_conflicting_stores": len(conflicts), "median_price_b": price_by_code.get("B", np.nan),
            "median_price_c": price_by_code.get("C", np.nan), "architecture_score": architecture_score,
            "rank_status": "outcome_blind_candidate_not_frozen",
        })
    return pd.DataFrame(records).sort_values(["architecture_score", "bc_stores"], ascending=False).reset_index(drop=True)


def one_week_support(rows: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Add adjacent-week B/C occurrence counts; zero is evidence for one-week support."""
    data = rows.copy()
    data["sale_code"] = data.SALE.fillna("").astype(str).str.strip().str.upper()
    key = data.loc[data.sale_code.isin(["B", "C"])].groupby(["UPC", "WEEK"]).STORE.nunique()
    output = candidate.copy()
    output["prior_week_bc_stores"] = [int(key.get((row.upc, row.week - 1), 0)) for row in output.itertuples()]
    output["next_week_bc_stores"] = [int(key.get((row.upc, row.week + 1), 0)) for row in output.itertuples()]
    output["one_week_price_side_support"] = output.prior_week_bc_stores.eq(0) & output.next_week_bc_stores.eq(0)
    return output


def store_assignment(rows: pd.DataFrame, *, upcs: Iterable[int], week: int) -> pd.DataFrame:
    """Jointly classify stores across supplied same-brand/size UPC variants."""
    targets = {int(upc) for upc in upcs}
    data = rows.loc[rows.WEEK.eq(week) & rows.UPC.isin(targets)].copy()
    data["sale_code"] = data.SALE.fillna("").astype(str).str.strip().str.upper()
    stores = sorted(set(data.STORE.dropna().astype(int)))
    output: list[dict[str, object]] = []
    for store in stores:
        group = data.loc[data.STORE.eq(store)]
        valid = group.loc[pd.to_numeric(group.OK, errors="coerce").eq(1)]
        codes = set(valid.sale_code).intersection({"B", "C"})
        if codes == {"B"}:
            assignment, confidence = "Bonus Buy", "strict_confirmed"
        elif codes == {"C"}:
            assignment, confidence = "Coupon", "strict_confirmed"
        elif codes:
            assignment, confidence = "unresolved", "conflicting_B_C_codes"
        else:
            assignment, confidence = "unresolved", "no_confirming_B_or_C_code"
        output.append({
            "store": store, "assignment": assignment, "assignment_confidence": confidence,
            "target_upc_count": int(group.UPC.nunique()), "valid_rows": int(len(valid)),
            "invalid_ok_zero_rows": int(pd.to_numeric(group.OK, errors="coerce").eq(0).sum()),
            "missing_sale_rows": int(valid.sale_code.eq("").sum()),
            "bonus_buy_rows": int(valid.sale_code.eq("B").sum()), "coupon_rows": int(valid.sale_code.eq("C").sum()),
        })
    return pd.DataFrame(output)


def price_summary(rows: pd.DataFrame, *, upcs: Iterable[int], week: int) -> pd.DataFrame:
    data = rows.loc[rows.WEEK.eq(week) & rows.UPC.isin(set(map(int, upcs)))].copy()
    data["sale_code"] = data.SALE.fillna("").astype(str).str.strip().str.upper()
    return data.groupby("sale_code", dropna=False, as_index=False).agg(
        rows=("STORE", "size"), stores=("STORE", "nunique"), price_min=("PRICE", "min"),
        price_median=("PRICE", "median"), price_max=("PRICE", "max"), quantities=("QTY", lambda values: ";".join(map(str, sorted(pd.Series(values).dropna().unique())))),
    )
