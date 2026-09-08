"""Outcome-blind contracts for the Dhar and Hoch (1996) search."""

from __future__ import annotations

import pandas as pd
import pytest

from dynamic_promotion_planning.dhar_hoch_reconstruction import (
    DharOutcomeFirewallError,
    assert_dhar_outcome_blind,
    one_week_support,
    scan_upc_week_architecture,
    store_assignment,
    upc_week_architecture,
)


def _synthetic_rows() -> pd.DataFrame:
    rows = []
    for store in range(1, 87):
        rows.append({"STORE": store, "UPC": 101, "WEEK": 177, "QTY": 1, "PRICE": 1.99, "SALE": "B" if store <= 43 else "C", "OK": 1})
    # Adjacent non-treatment observations prove the candidate's one-week support.
    for week in (176, 178):
        rows.append({"STORE": 1, "UPC": 101, "WEEK": week, "QTY": 1, "PRICE": 2.84, "SALE": "", "OK": 1})
    return pd.DataFrame(rows)


def test_identifiable_synthetic_43_43_pattern_is_ranked_without_outcomes():
    ranked = one_week_support(_synthetic_rows(), upc_week_architecture(_synthetic_rows()))
    candidate = ranked.loc[(ranked.upc == 101) & (ranked.week == 177)].iloc[0]
    assert (candidate.b_stores, candidate.c_stores, candidate.bc_stores) == (43, 43, 86)
    assert candidate.one_week_price_side_support
    assignments = store_assignment(_synthetic_rows(), upcs=[101], week=177)
    assert assignments.assignment.value_counts().to_dict() == {"Bonus Buy": 43, "Coupon": 43}


def test_conflicting_or_missing_codes_remain_unresolved():
    rows = _synthetic_rows()
    rows = pd.concat([rows, pd.DataFrame([{"STORE": 1, "UPC": 102, "WEEK": 177, "QTY": 1, "PRICE": 1.99, "SALE": "C", "OK": 1}])], ignore_index=True)
    assignments = store_assignment(rows, upcs=[101, 102], week=177)
    assert assignments.loc[assignments.store.eq(1), "assignment"].item() == "unresolved"
    assert assignments.loc[assignments.store.eq(1), "assignment_confidence"].item() == "conflicting_B_C_codes"


def test_move_is_rejected_before_candidate_discovery():
    with pytest.raises(DharOutcomeFirewallError, match="MOVE"):
        assert_dhar_outcome_blind(["STORE", "UPC", "MOVE"])


def test_streaming_scanner_recovers_synthetic_candidate_without_loading_outcomes(tmp_path):
    path = tmp_path / "movement.csv"
    _synthetic_rows().to_csv(path, index=False)
    ranked = scan_upc_week_architecture(path, [176, 177, 178], chunksize=10)
    candidate = ranked.loc[(ranked.upc == 101) & (ranked.week == 177)].iloc[0]
    assert (candidate.b_stores, candidate.c_stores, candidate.bc_stores) == (43, 43, 86)
