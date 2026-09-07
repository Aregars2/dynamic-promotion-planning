"""Build an outcome-blind inventory of the raw Dominick's movement archives.

This is the first stage of the experimental-reconstruction workflow.  It reads
only fields permitted before the reconstruction freeze: store, UPC, week,
price, deal indicator, and validity flag.  In particular, it never loads MOVE,
QTY, or PROFIT.

Usage
-----
python scripts/experimental_reconstruction/phase0_data_audit.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from dynamic_promotion_planning.experimental_reconstruction import (
    OUTCOME_FIELDS,
    RECONSTRUCTION_FIELDS,
    assert_outcome_blind,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "experimental_reconstruction" / "phase0"

# These are the only movement-file fields permitted during reconstruction.
ALLOWED_FIELDS = RECONSTRUCTION_FIELDS
FORBIDDEN_OUTCOME_FIELDS = OUTCOME_FIELDS


@dataclass
class CategoryAudit:
    category_code: str
    archive: str
    member: str
    schema: tuple[str, ...]
    rows: int = 0
    stores: set[int] = field(default_factory=set)
    upcs: set[int] = field(default_factory=set)
    first_week: int | None = None
    last_week: int | None = None
    missing_store: int = 0
    missing_upc: int = 0
    missing_week: int = 0
    missing_price: int = 0
    missing_sale: int = 0
    missing_ok: int = 0
    invalid_ok_zero: int = 0
    store_rows: dict[int, int] = field(default_factory=dict)
    store_first_week: dict[int, int] = field(default_factory=dict)
    store_last_week: dict[int, int] = field(default_factory=dict)
    store_valid_rows: dict[int, int] = field(default_factory=dict)
    store_observed_price_rows: dict[int, int] = field(default_factory=dict)


def category_code(member: str) -> str:
    """Return the archive's transparent three-letter movement-category code."""
    stem = Path(member).stem.lower()
    return stem[1:] if stem.startswith("w") else stem


def archive_member(archive: Path) -> str:
    with ZipFile(archive) as handle:
        members = [name for name in handle.namelist() if name.lower().endswith(".csv")]
    if len(members) != 1:
        raise ValueError(f"Expected one CSV in {archive.name}; found {members!r}")
    return members[0]


def update_minimum(current: int | None, values: pd.Series) -> int | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return current
    value = int(values.min())
    return value if current is None else min(current, value)


def update_maximum(current: int | None, values: pd.Series) -> int | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return current
    value = int(values.max())
    return value if current is None else max(current, value)


def audit_archive(archive: Path, chunk_rows: int) -> CategoryAudit:
    """Scan one archive without loading any demand or profit outcome fields."""
    member = archive_member(archive)
    with ZipFile(archive) as handle:
        with handle.open(member) as raw_file:
            header = raw_file.readline().decode("latin-1").strip().split(",")
    schema = tuple(header)
    assert_outcome_blind(ALLOWED_FIELDS)
    forbidden = FORBIDDEN_OUTCOME_FIELDS.intersection(schema)
    if not forbidden.issuperset({"MOVE", "QTY", "PROFIT"}):
        raise ValueError(f"Unexpected movement schema in {archive.name}: {schema!r}")
    missing_allowed = set(ALLOWED_FIELDS).difference(schema)
    if missing_allowed:
        raise ValueError(f"Missing permitted audit fields in {archive.name}: {missing_allowed}")

    audit = CategoryAudit(
        category_code=category_code(member), archive=archive.name, member=member, schema=schema
    )
    # The explicit allow-list is the outcome firewall. Do not replace it with
    # a full-file read followed by column selection: that would violate the
    # reconstruction protocol even if downstream calculations ignored outcomes.
    reader = pd.read_csv(
        archive,
        compression="zip",
        usecols=list(ALLOWED_FIELDS),
        chunksize=chunk_rows,
        low_memory=False,
    )
    for chunk in reader:
        audit.rows += len(chunk)
        for field, attribute in (
            ("STORE", "missing_store"),
            ("UPC", "missing_upc"),
            ("WEEK", "missing_week"),
            ("PRICE", "missing_price"),
            ("SALE", "missing_sale"),
            ("OK", "missing_ok"),
        ):
            setattr(audit, attribute, getattr(audit, attribute) + int(chunk[field].isna().sum()))

        store = pd.to_numeric(chunk["STORE"], errors="coerce")
        upc = pd.to_numeric(chunk["UPC"], errors="coerce")
        week = pd.to_numeric(chunk["WEEK"], errors="coerce")
        price = pd.to_numeric(chunk["PRICE"], errors="coerce")
        valid = pd.to_numeric(chunk["OK"], errors="coerce")
        audit.stores.update(store.dropna().astype(int).unique())
        audit.upcs.update(upc.dropna().astype(int).unique())
        audit.first_week = update_minimum(audit.first_week, week)
        audit.last_week = update_maximum(audit.last_week, week)
        audit.invalid_ok_zero += int((valid == 0).sum())

        frame = pd.DataFrame({"store": store, "week": week, "valid": valid, "price": price})
        frame = frame.dropna(subset=["store"])
        frame["store"] = frame["store"].astype(int)
        for store_id, group in frame.groupby("store", sort=False):
            audit.store_rows[store_id] = audit.store_rows.get(store_id, 0) + len(group)
            audit.store_first_week[store_id] = update_minimum(
                audit.store_first_week.get(store_id), group["week"]
            )
            audit.store_last_week[store_id] = update_maximum(
                audit.store_last_week.get(store_id), group["week"]
            )
            audit.store_valid_rows[store_id] = audit.store_valid_rows.get(store_id, 0) + int(
                (group["valid"] == 1).sum()
            )
            audit.store_observed_price_rows[store_id] = audit.store_observed_price_rows.get(
                store_id, 0
            ) + int(group["price"].notna().sum())
    return audit


def inventory_rows(audits: Iterable[CategoryAudit], raw_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in audits:
        rows.append(
            {
                "category_code": audit.category_code,
                "source_archive": audit.archive,
                "source_member": audit.member,
                "archive_bytes": (raw_dir / audit.archive).stat().st_size,
                "schema": ";".join(audit.schema),
                "rows": audit.rows,
                "n_stores": len(audit.stores),
                "n_upcs": len(audit.upcs),
                "first_week": audit.first_week,
                "last_week": audit.last_week,
                "missing_store_rows": audit.missing_store,
                "missing_upc_rows": audit.missing_upc,
                "missing_week_rows": audit.missing_week,
                "missing_price_rows": audit.missing_price,
                "missing_sale_rows": audit.missing_sale,
                "missing_ok_rows": audit.missing_ok,
                "invalid_ok_zero_rows": audit.invalid_ok_zero,
            }
        )
    return rows


def coverage_rows(audits: Iterable[CategoryAudit]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in audits:
        for store_id in sorted(audit.store_rows):
            rows.append(
                {
                    "category_code": audit.category_code,
                    "store": store_id,
                    "rows": audit.store_rows[store_id],
                    "first_week": audit.store_first_week.get(store_id),
                    "last_week": audit.store_last_week.get(store_id),
                    "valid_ok_rows": audit.store_valid_rows.get(store_id, 0),
                    "observed_price_rows": audit.store_observed_price_rows.get(store_id, 0),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    args = parser.parse_args()

    archives = sorted(args.raw_dir.glob("*.zip"))
    if len(archives) != 28:
        raise ValueError(f"Expected all 28 public category archives; found {len(archives)}")
    audits = [audit_archive(archive, args.chunk_rows) for archive in archives]
    codes = [audit.category_code for audit in audits]
    if len(set(codes)) != len(codes):
        raise ValueError(f"Duplicate category codes inferred from archives: {codes!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory_rows(audits, args.raw_dir)).sort_values("category_code").to_csv(
        args.output_dir / "data_inventory.csv", index=False
    )
    pd.DataFrame(coverage_rows(audits)).sort_values(["category_code", "store"]).to_csv(
        args.output_dir / "store_category_coverage.csv", index=False
    )
    print(f"Audited {len(audits)} category archives without reading outcome fields.")
    print(f"Wrote {args.output_dir / 'data_inventory.csv'}")
    print(f"Wrote {args.output_dir / 'store_category_coverage.csv'}")


if __name__ == "__main__":
    main()
