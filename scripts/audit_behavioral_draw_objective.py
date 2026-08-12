"""Export a deterministic πN/πD draw-weighted schedule-objective audit."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dynamic_promotion_planning.policy import PlanningSpec, audit_draw_weighted_schedule_values, build_schedule_system


def main() -> None:
    planning = PlanningSpec(decision_horizon=2, washout_horizon=0, cooldown=0, max_promotions=1)
    draws = {"A": {"epsilon": np.array([1., 2.]), "gamma": np.array([.1, .5]), "psi": np.array([.2, .8]), "r": np.array([.3, .7]), "weights": np.array([.25, .75]), "base_demand": np.array([10., 20.]), "regular_price": np.array([2., 2.]), "unit_cost": np.array([1., 1.])}}
    profiles = {"A": {"baseline_demand": np.array([15., 15.]), "price_factor": np.ones(2), "cost_factor": np.ones(2)}}
    output = []
    for name, new_state in [("piD", True), ("piN", False)]:
        system = build_schedule_system(draws, profiles, {"A": (0., .2)}, planning, [1.], add_new_promotion_displacement=new_state)
        frame = audit_draw_weighted_schedule_values(system, draws, profiles, upc="A", alpha=1., capacity=1)
        frame.insert(0, "policy", name)
        output.append(frame)
    target = ROOT / "results" / "final" / "tables" / "behavioral_draw_weighted_schedule_audit.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.concat(output, ignore_index=True).to_csv(target, index=False)
    print(target)


if __name__ == "__main__":
    main()
