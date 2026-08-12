"""Fixed 36-versus-52-week terminal-tail sensitivity on a representative policy instance."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dynamic_promotion_planning.policy import (
    PlanningSpec, build_schedule_system, evaluate_schedule_map,
    run_three_policy_grid, schedule_signature,
)


def _inputs(tail: int):
    planning = PlanningSpec(decision_horizon=12, washout_horizon=tail, cooldown=2, max_promotions=3, discount_factor=1.0)
    draws, profiles, actions = {}, {}, {}
    for upc, base, psi, persistence, gamma in [("A", 14., .9, .60, .38), ("B", 11., .6, .75, .30)]:
        draws[upc] = {
            "epsilon": np.array([1.0, 1.8]), "gamma": np.array([gamma, gamma * 1.15]),
            "psi": np.array([psi, psi * 1.2]), "r": np.array([persistence * .8, persistence]),
            "weights": np.array([.4, .6]), "base_demand": np.array([base, base * 1.1]),
            "regular_price": np.array([2.2, 2.2]), "unit_cost": np.array([1.0, 1.0]),
        }
        profiles[upc] = {"baseline_demand": np.full(planning.evaluation_horizon, base), "price_factor": np.ones(planning.evaluation_horizon), "cost_factor": np.ones(planning.evaluation_horizon)}
        actions[upc] = (0.0, .10, .20)
    return planning, draws, profiles, actions


def _run(tail: int) -> tuple[dict, dict, dict, dict]:
    planning, draws, profiles, actions = _inputs(tail)
    dynamic = build_schedule_system(draws, profiles, actions, planning, [1.0])
    naive = build_schedule_system(draws, profiles, actions, planning, [1.0], add_new_promotion_displacement=False)
    result = run_three_policy_grid(dynamic, naive, draws, profiles, actions, [1.0], [1])
    schedules = result["schedules"][(1.0, 1)]
    details = {name: evaluate_schedule_map(schedule, draws, profiles, planning, 1.0) for name, schedule in schedules.items()}
    return result["results"].iloc[0].to_dict(), schedules, details, {"planning": planning, "draws": draws, "profiles": profiles}


def main() -> None:
    r36, s36, d36, _ = _run(36)
    r52, s52, d52, _ = _run(52)
    values = {"piM": "value_piM", "piN": "value_piN", "piD": "value_piD"}
    keys = {"piM": "myopic", "piN": "naive_dynamic", "piD": "dynamic"}
    rows = []
    for policy, value_col in values.items():
        key = keys[policy]
        before, after = float(r36[value_col]), float(r52[value_col])
        rows.append({"metric_type": "policy", "metric": policy, "value_w36": before, "value_w52": after,
                     "absolute_change": after - before, "relative_change": (after - before) / abs(before),
                     "max_remaining_state_w36": max(d36[key]["terminal_state"].values()),
                     "max_remaining_state_w52": max(d52[key]["terminal_state"].values()),
                     "calendar_changed": schedule_signature(s36[key]) != schedule_signature(s52[key])})
    for metric in ["delta_plan", "delta_disp", "delta_total"]:
        before, after = float(r36[metric]), float(r52[metric])
        rows.append({"metric_type": "decomposition", "metric": metric, "value_w36": before, "value_w52": after,
                     "absolute_change": after - before, "relative_change": (after - before) / max(abs(before), 1e-12),
                     "max_remaining_state_w36": np.nan, "max_remaining_state_w52": np.nan,
                     "calendar_changed": np.nan})
    ranking36 = tuple(sorted(values, key=lambda name: r36[values[name]], reverse=True))
    ranking52 = tuple(sorted(values, key=lambda name: r52[values[name]], reverse=True))
    rows.append({"metric_type": "comparison", "metric": "policy_ranking", "value_w36": np.nan, "value_w52": np.nan,
                 "absolute_change": np.nan, "relative_change": np.nan,
                 "max_remaining_state_w36": np.nan, "max_remaining_state_w52": np.nan,
                 "calendar_changed": ranking36 != ranking52,
                 "ranking_w36": ">".join(ranking36), "ranking_w52": ">".join(ranking52)})
    report = pd.DataFrame(rows)
    target = ROOT / "results" / "final" / "tables" / "terminal_tail_w36_w52_sensitivity.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(target, index=False)
    assert np.isclose(r36["delta_plan"] + r36["delta_disp"], r36["delta_total"])
    assert np.isclose(r52["delta_plan"] + r52["delta_disp"], r52["delta_total"])
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
