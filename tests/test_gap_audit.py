import numpy as np

from dynamic_promotion_planning.gap_audit import relative_gap, transition_neighborhoods


def _schedule(value: float) -> dict[str, np.ndarray]:
    return {"A": np.array([value, 0.0])}


def test_transition_neighborhoods_marks_both_sides_and_policy() -> None:
    schedules = {
        (0.0, 2): {"myopic": _schedule(0), "naive_dynamic": _schedule(0), "dynamic": _schedule(0)},
        (0.01, 2): {"myopic": _schedule(0), "naive_dynamic": _schedule(1), "dynamic": _schedule(0)},
    }
    result = transition_neighborhoods(schedules, [0.0, 0.01], [2])
    assert result == {(0.0, 2): "piN", (0.01, 2): "piN"}


def test_relative_gap_uses_stable_denominator() -> None:
    assert relative_gap(0.0, 0.25) == 0.25
    assert relative_gap(-100.0, 2.0) == 0.02
