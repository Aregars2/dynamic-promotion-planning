"""Synthetic contracts for outcome-blind experimental-reconstruction tooling."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from dynamic_promotion_planning.experimental_reconstruction import (
    FreezeGateError,
    OutcomeFirewallError,
    WeekDecoder,
    build_freeze_manifest,
    candidate_diagnostics,
    cross_category_agreement,
    read_historical_evidence,
    require_passed_freeze,
)


def _evidence_rows() -> list[dict[str, object]]:
    base = {
        "source": "archival design memorandum", "locator": "p. 2", "documented_start_date": "1992-07-23",
        "documented_end_date": "1992-11-14", "assignment": "", "confidence": "high",
        "independent_of_outcomes": "true", "count_implied": "false",
    }
    return [
        dict(base, evidence_id="s1", study="Study 1", evidence_type="calendar_window", unit_type="window", unit_id="s1", frame_status=""),
        dict(base, evidence_id="s2", study="Study 2", evidence_type="calendar_window", unit_type="window", unit_id="s2", frame_status=""),
    ]


def test_week_decoder_and_external_constraints_are_deterministic(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    pd.DataFrame(_evidence_rows()).to_csv(evidence_path, index=False)
    constraints = read_historical_evidence(evidence_path)
    assert WeekDecoder().week_bounds(1) == (date(1989, 9, 14), date(1989, 9, 20))
    assert constraints.admissible_starts("Study 1", 16) == set(range(150, 152))
    candidates = pd.DataFrame({"candidate_start_week": [149, 150, 151], "score": [4.0, 3.0, 2.0]})
    result = candidate_diagnostics(candidates, study="Study 1", constraints=constraints, window_weeks=16)
    assert result.historically_admissible.tolist() == [False, True, True]


def test_outcome_firewall_rejects_direct_and_evidence_leakage(tmp_path):
    from dynamic_promotion_planning.experimental_reconstruction import assert_outcome_blind

    with pytest.raises(OutcomeFirewallError, match="MOVE"):
        assert_outcome_blind(["STORE", "MOVE"])
    rows = _evidence_rows()
    rows[0]["source"] = "published profit outcome table"
    path = tmp_path / "bad.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(OutcomeFirewallError):
        read_historical_evidence(path)


def test_cross_category_agreement_is_diagnostic_not_assignment():
    evidence = pd.DataFrame(
        {"store": [1, 1, 1, 2, 2], "category_code": ["a", "b", "c", "a", "b"], "log_price_shift": [0.2, 0.1, -0.1, -0.2, -0.1]}
    )
    result = cross_category_agreement(evidence)
    assert set(result.interpretation) == {"price_side_agreement_not_treatment_assignment"}
    assert result.loc[result.store.eq(1), "sign_agreement_share"].item() == pytest.approx(2 / 3)


def test_freeze_passes_only_for_independent_complete_frame_and_phase5_is_guarded(tmp_path):
    evidence = pd.DataFrame(_evidence_rows())
    categories = [dict(_evidence_rows()[0], evidence_id=f"c{i}", evidence_type="membership", unit_type="category", unit_id=str(i), frame_status="in_frame") for i in range(26)]
    stores = [dict(_evidence_rows()[0], evidence_id=f"s{i}", evidence_type="membership", unit_type="store", unit_id=str(i), frame_status="in_frame") for i in range(86)]
    evidence = pd.concat([evidence, pd.DataFrame(categories + stores)], ignore_index=True)
    evidence["independent_of_outcomes"] = True
    evidence["count_implied"] = False
    payload = tmp_path / "labels.csv"
    payload.write_text("synthetic labels\n", encoding="utf-8")
    manifest = build_freeze_manifest(output_dir=tmp_path, files=[payload], protocol_version="synthetic-v1", evidence=evidence)
    assert manifest["freeze_status"] == "passed"
    assert require_passed_freeze(tmp_path / "phase4_freeze_manifest.json")["freeze_status"] == "passed"
    manifest["freeze_status"] = "blocked"
    (tmp_path / "blocked.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FreezeGateError):
        require_passed_freeze(tmp_path / "blocked.json")


def test_ambiguous_synthetic_frame_remains_blocked(tmp_path):
    evidence = pd.DataFrame(_evidence_rows())
    payload = tmp_path / "labels.csv"
    payload.write_text("ambiguous\n", encoding="utf-8")
    manifest = build_freeze_manifest(output_dir=tmp_path, files=[payload], protocol_version="synthetic-v1", evidence=evidence)
    assert manifest["freeze_status"] == "blocked"
