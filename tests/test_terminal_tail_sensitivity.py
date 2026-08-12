from __future__ import annotations

from scripts.audit_terminal_tail import _run


def test_fixed_terminal_tail_sensitivity_preserves_three_policy_accounting():
    result36, _, details36, _ = _run(36)
    result52, _, details52, _ = _run(52)
    for result in (result36, result52):
        assert abs(result["delta_plan"] + result["delta_disp"] - result["delta_total"]) < 1e-10
    assert max(details52["dynamic"]["terminal_state"].values()) < max(details36["dynamic"]["terminal_state"].values())
