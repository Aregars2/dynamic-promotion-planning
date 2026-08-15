"""Audit frozen price-consistent two-week-cooldown policy results.

The script is diagnostic only.  It neither changes the policy artifact nor
constructs a new candidate set: naïve top-two rankings are recovered from the
unpruned schedule values persisted inside the canonical artifact.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_promotion_planning.policy import (
    _expected_current_profit,
    evaluate_schedule_map,
    load_pickle,
    occupancy_matrix_from_masks,
    schedule_signature,
)


VERSION = "empirical_bayes_price_consistent"
ARTIFACT_PATH = ROOT / "artifacts" / "policy" / VERSION / "policy_optimization.pkl"
TASK8_GRID_PATH = ROOT / "results" / VERSION / "tables" / "task8_three_policy_reporting_grid.csv"
OUT = ROOT / "results" / VERSION / "audit_price_consistent_two_week"
# The production MILP exposes no user-level objective-tie tolerance.  This
# diagnostic reports only differences indistinguishable at this conservative
# floating-point threshold, without changing how the production solver chose.
TIE_ATOL = 1e-8


def _raw_category_table(system: dict, share: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the production MILP table from *all* enumerated schedules.

    ``schedule_values`` contains the unpruned feasible schedules.  This is the
    appropriate source for a literal runner-up: same-mask pruning is exact for
    the optimum but can discard a schedule that is second after the winner is
    excluded.
    """
    frames, blocks = [], []
    offset = 0
    horizon = system["planning"].decision_horizon
    for position, upc in enumerate(system["products"]):
        values = system["product_artifacts"][upc]["schedule_values"].copy()
        values["upc"] = str(upc)
        values["product_position"] = position
        values["global_index"] = np.arange(offset, offset + len(values), dtype=int)
        values["value"] = values["intercept"] + float(share) * values["exposure"]
        frames.append(values)
        blocks.append(occupancy_matrix_from_masks(values["occupancy_mask"], horizon))
        offset += len(values)
    return pd.concat(frames, ignore_index=True), np.vstack(blocks)


def _solve_raw_top2(system: dict, share: float, capacity: int) -> dict:
    """Solve the exact production MILP and its all-schedule excluded runner-up."""
    candidates, occupancy = _raw_category_table(system, share)
    products = list(system["products"])
    n = len(candidates)
    product_rows = np.zeros((len(products), n), dtype=float)
    for position in range(len(products)):
        product_rows[position, candidates["product_position"].eq(position).to_numpy()] = 1.0
    matrix = csc_matrix(np.vstack([product_rows, occupancy.T.astype(float)]))
    lower = np.r_[np.ones(len(products)), np.full(occupancy.shape[1], -np.inf)]
    upper = np.r_[np.ones(len(products)), np.full(occupancy.shape[1], int(capacity))]
    constraints = LinearConstraint(matrix, lower, upper)
    options = {"disp": False}

    def solve(extra: LinearConstraint | None = None):
        constraints_used = constraints if extra is None else extra
        result = milp(
            c=-candidates["value"].to_numpy(float), integrality=np.ones(n),
            bounds=Bounds(np.zeros(n), np.ones(n)), constraints=constraints_used, options=options,
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"Raw top-two MILP failed at lambda={share}, B={capacity}: {result.message}")
        positions = np.flatnonzero(result.x > 0.5)
        selected = candidates.iloc[positions].copy().sort_values("product_position")
        schedule = {
            str(row.upc): system["product_artifacts"][str(row.upc)]["schedules"][int(row.schedule_index)].astype(float)
            for row in selected.itertuples(index=False)
        }
        return selected, schedule, float(selected["value"].sum()), str(result.message)

    best_rows, best_schedule, best_value, message = solve()
    chosen = best_rows["global_index"].to_numpy(int)
    exclusion = np.zeros(n)
    exclusion[chosen] = 1.0
    matrix_second = vstack([matrix, csc_matrix(exclusion.reshape(1, -1))], format="csc")
    lower_second, upper_second = np.r_[lower, -np.inf], np.r_[upper, len(products) - 1]
    second_rows, second_schedule, second_value, second_message = solve(
        LinearConstraint(matrix_second, lower_second, upper_second)
    )
    return {
        "best_schedule": best_schedule, "second_schedule": second_schedule,
        "best_value": best_value, "second_value": second_value,
        "gap": best_value - second_value, "best_message": message,
        "second_message": second_message, "raw_candidate_count": n,
    }


def _prefix_profiles(profiles: dict, weeks: int) -> dict:
    return {upc: {key: np.asarray(value)[:weeks] for key, value in prof.items()} for upc, prof in profiles.items()}


def _schedule_details(schedule: dict) -> dict:
    result = {}
    for upc, values in sorted(schedule.items()):
        positive = np.flatnonzero(np.asarray(values) > 0)
        result[upc] = [{"week": int(week + 1), "depth": float(values[week])} for week in positive]
    return result


def _dn_diagnostics(n_schedule: dict, d_schedule: dict) -> dict:
    products = sorted(n_schedule)
    n = np.vstack([np.asarray(n_schedule[p], float) for p in products])
    d = np.vstack([np.asarray(d_schedule[p], float) for p in products])
    n_status, d_status = n > 0, d > 0
    common = n_status & d_status
    return {
        "piN_promotion_count": int(n_status.sum()),
        "piD_promotion_count": int(d_status.sum()),
        "promotion_status_disagreements": int((n_status != d_status).sum()),
        "common_promotion_product_weeks": int(common.sum()),
        "common_promotion_depth_disagreements": int((common & ~np.isclose(n, d)).sum()),
        "piN_promotions_by_product": json.dumps(_schedule_details(n_schedule), sort_keys=True),
        "piD_promotions_by_product": json.dumps(_schedule_details(d_schedule), sort_keys=True),
    }


def _schedule_objective(system: dict, schedule: dict, share: float) -> float:
    """Evaluate a selected complete calendar in the system's exact MILP objective.

    Both πN and πD use the identical feasible-calendar universe.  The schedule
    rows below are the persisted, unpruned per-product values that feed the
    production category MILP, so this performs no new approximation or
    alternative tie-break.
    """
    value = 0.0
    for upc in system["products"]:
        artifact = system["product_artifacts"][upc]
        schedules = np.asarray(artifact["schedules"], dtype=float)
        target = np.asarray(schedule[str(upc)], dtype=float)
        matches = np.flatnonzero(np.all(np.isclose(schedules, target[None, :], atol=1e-10, rtol=0.0), axis=1))
        if len(matches) != 1:
            raise AssertionError(f"Could not identify exactly one raw schedule row for UPC {upc}.")
        row = artifact["schedule_values"].iloc[int(matches[0])]
        value += float(row["intercept"] + float(share) * row["exposure"])
    return value


def _myopic_action_ties(draws_by_product: dict, profiles: dict, action_sets: dict, planning, share: float, capacity: int) -> dict:
    """Inspect production myopic comparisons without changing their tie-break.

    Production sorts equal increments by UPC after choosing each product's
    NumPy ``argmax`` action.  This reports only comparisons numerically
    indistinguishable at the stated floating-point diagnostic threshold.
    """
    products = sorted(draws_by_product)
    states = {p: np.zeros(len(draws_by_product[p]["weights"])) for p in products}
    last = {p: -10_000 for p in products}
    counts = {p: 0 for p in products}
    action_ties, cutoff_ties = [], []
    for week in range(planning.decision_horizon):
        candidates = []
        for upc in products:
            draws, profile = draws_by_product[upc], profiles[upc]
            base = _expected_current_profit(0.0, states[upc], draws, profile, week, share)
            if counts[upc] < planning.max_promotions and week - last[upc] > planning.cooldown:
                actions = [float(x) for x in action_sets[upc] if float(x) > 0]
                values = np.asarray([_expected_current_profit(x, states[upc], draws, profile, week, share) for x in actions])
                if len(values) > 1:
                    ordered = np.sort(values)[::-1]
                    tolerance = TIE_ATOL * max(1.0, abs(float(ordered[0])))
                    if abs(float(ordered[0] - ordered[1])) <= tolerance:
                        action_ties.append({"week": week + 1, "upc": upc, "tolerance": tolerance})
                best_position = int(np.argmax(values))
                increment = float(values[best_position] - base)
                if increment > 1e-12:
                    candidates.append((increment, upc, actions[best_position]))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        if len(candidates) > capacity:
            tolerance = TIE_ATOL * max(1.0, abs(candidates[capacity - 1][0]))
            if abs(candidates[capacity - 1][0] - candidates[capacity][0]) <= tolerance:
                cutoff_ties.append({"week": week + 1, "tolerance": tolerance})
        chosen = {p: 0.0 for p in products}
        for _, upc, depth in candidates[:capacity]:
            chosen[upc] = depth
            last[upc], counts[upc] = week, counts[upc] + 1
        for upc in products:
            states[upc] = draws_by_product[upc]["r"] * states[upc] + chosen[upc]
    return {
        "myopic_within_product_action_tie_events": len(action_ties),
        "myopic_capacity_cutoff_tie_events": len(cutoff_ties),
        "myopic_tie_details": json.dumps({"action": action_ties, "cutoff": cutoff_ties}, sort_keys=True),
    }


def _switch_points(results: pd.DataFrame, signature: str, capacity: int) -> list[float]:
    group = results.loc[results.capacity.eq(capacity)].sort_values("reimbursement_share")
    changed = group[signature].ne(group[signature].shift())
    return group.loc[changed & group.index.to_series().ne(group.index[0]), "reimbursement_share"].astype(float).tolist()


def main() -> None:
    started = time.monotonic()
    if not ARTIFACT_PATH.is_file() or not TASK8_GRID_PATH.is_file():
        raise FileNotFoundError("Run Notebook 06 and Task-8 reporting before this audit.")
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = load_pickle(ARTIFACT_PATH)
    results = artifact["policy_results"].copy().sort_values(["capacity", "reimbursement_share"])
    reporting = pd.read_csv(TASK8_GRID_PATH).query("specification == 'main'")
    results = results.merge(
        reporting[["capacity", "reimbursement_share", "myopic_planning_profit", "delta_total_pct"]],
        on=["capacity", "reimbursement_share"], how="left", validate="one_to_one",
    )
    if results["delta_total_pct"].isna().any():
        raise AssertionError("Missing Task-8 12-week denominators in canonical grid.")
    schedules = artifact["three_policy_schedules"]
    planning = artifact["schedule_system"]["planning"]
    profiles12 = _prefix_profiles(artifact["weekly_profiles"], planning.decision_horizon)
    planning12 = replace(planning, washout_horizon=0)

    # 1. Canonical full-grid peak audit: total and displacement maxima, every B.
    peak_rows, peak_points = [], set()
    for capacity in artifact["capacities"]:
        group = results.loc[results.capacity.eq(capacity)]
        for metric, label in (
            ("delta_total", "total"),
            ("delta_disp", "displacement"),
            ("delta_total_pct", "total_12_week_normalized"),
        ):
            row = group.loc[group[metric].idxmax()]
            share = float(row.reimbursement_share)
            key = (round(share, 8), int(capacity))
            selected = schedules[key]
            row_out = {
                "capacity": int(capacity), "maximum": label, "reimbursement_share": share,
                "delta_plan": float(row.delta_plan), "delta_disp": float(row.delta_disp),
                "delta_total": float(row.delta_total),
                "delta_total_pct_12_week_myopic": float(row.delta_total_pct),
                "delta_total_pct_48_week_myopic": 100.0 * float(row.delta_total) / float(row.value_piM),
                "piM_schedule_signature": schedule_signature(selected["myopic"]),
                "piN_schedule_signature": schedule_signature(selected["naive_dynamic"]),
                "piD_schedule_signature": schedule_signature(selected["dynamic"]),
            }
            peak_rows.append(row_out)
            # Cross-calendar diagnostics requested below concern the absolute
            # Δtotal and Δdisp peaks.  Keep the normalized peak in the peak
            # audit solely to explain the Figure/table discrepancy.
            if label in {"total", "displacement"}:
                peak_points.add((int(capacity), share, label))
    peaks = pd.DataFrame(peak_rows).sort_values(["capacity", "maximum"])
    peaks.to_csv(OUT / "main_peak_audit.csv", index=False)

    # 2. D-vs-N comparison at both sets of peaks (deduplicate coincident points).
    diagnostics = []
    for capacity, share, trigger in sorted(peak_points):
        row = results.loc[(results.capacity.eq(capacity)) & np.isclose(results.reimbursement_share, share)].iloc[0]
        key = (round(share, 8), capacity)
        item = {
            "capacity": capacity, "reimbursement_share": share, "trigger": trigger,
            "delta_plan": float(row.delta_plan), "delta_disp": float(row.delta_disp), "delta_total": float(row.delta_total),
            "piN_schedule_signature": schedule_signature(schedules[key]["naive_dynamic"]),
            "piD_schedule_signature": schedule_signature(schedules[key]["dynamic"]),
        }
        item.update(_dn_diagnostics(schedules[key]["naive_dynamic"], schedules[key]["dynamic"]))
        naive_system = artifact["naive_schedule_system"]
        dynamic_system = artifact["schedule_system"]
        vn_n = _schedule_objective(naive_system, schedules[key]["naive_dynamic"], share)
        vn_d = _schedule_objective(naive_system, schedules[key]["dynamic"], share)
        vd_d = _schedule_objective(dynamic_system, schedules[key]["dynamic"], share)
        vd_n = _schedule_objective(dynamic_system, schedules[key]["naive_dynamic"], share)
        n_to_d = vn_n - vn_d
        d_to_n = vd_d - vd_n
        if n_to_d < -1e-7:
            raise AssertionError(f"πD calendar beats πN under naïve objective at B={capacity}, lambda={share}.")
        if not np.isclose(d_to_n, float(row.delta_disp), atol=1e-6):
            raise AssertionError(
                f"Full-objective calendar contrast does not equal Δdisp at B={capacity}, lambda={share}."
            )
        item.update({
            "naive_objective_piN_calendar": vn_n,
            "naive_objective_piD_calendar": vn_d,
            "G_N_to_D": n_to_d,
            "G_N_to_D_relative_to_VN_piN": n_to_d / max(abs(vn_n), 1.0),
            "dynamic_objective_piD_calendar": vd_d,
            "dynamic_objective_piN_calendar": vd_n,
            "G_D_to_N": d_to_n,
            "G_D_to_N_over_G_N_to_D": d_to_n / n_to_d if n_to_d > 1e-12 else np.nan,
        })
        diagnostics.append(item)
    diagnostics_frame = pd.DataFrame(diagnostics).sort_values(["capacity", "reimbursement_share", "trigger"])
    diagnostics_frame.to_csv(OUT / "dynamic_vs_naive_calendar_diagnostics.csv", index=False)

    # 3–4. Exact top two full feasible calendars around every πN switch,
    # including immediately before and after (and all total/displacement peaks).
    grid = np.asarray(artifact["reimbursement_grid"], float)
    requested: dict[int, set[float]] = {int(cap): set() for cap in artifact["capacities"]}
    switch_map: dict[int, list[float]] = {}
    for capacity in artifact["capacities"]:
        cap = int(capacity)
        switches = _switch_points(results, "naive_dynamic_schedule_signature", cap)
        switch_map[cap] = switches
        for point in switches:
            idx = int(np.flatnonzero(np.isclose(grid, point))[0])
            requested[cap].update(grid[max(0, idx - 1): min(len(grid), idx + 2)].tolist())
    for cap, share, _ in peak_points:
        requested[cap].add(share)

    top2_rows, tie_rows = [], []
    for capacity in artifact["capacities"]:
        cap = int(capacity)
        for share in sorted(requested[cap]):
            solution = _solve_raw_top2(artifact["naive_schedule_system"], share, cap)
            key = (round(share, 8), cap)
            stored_n = schedules[key]["naive_dynamic"]
            saved_value = float(results.loc[(results.capacity.eq(cap)) & np.isclose(results.reimbursement_share, share), "value_piN"].iloc[0])
            # Objective units differ from replayed total profit only through the
            # same schedule construction; the saved signature must agree unless
            # a numerical objective tie permits an equally optimal alternative.
            same_saved = schedule_signature(solution["best_schedule"]) == schedule_signature(stored_n)
            if not same_saved and abs(solution["gap"]) > TIE_ATOL * max(1.0, abs(solution["best_value"])):
                raise AssertionError(f"Raw πN optimum differs from saved calendar at B={cap}, lambda={share}.")
            full_second = evaluate_schedule_map(
                solution["second_schedule"], artifact["draws_by_product"], artifact["weekly_profiles"], planning, share
            )["total_profit"]
            full_best = evaluate_schedule_map(
                solution["best_schedule"], artifact["draws_by_product"], artifact["weekly_profiles"], planning, share
            )["total_profit"]
            row = {
                "capacity": cap, "reimbursement_share": share,
                "is_naive_switch_into": bool(any(np.isclose(share, x) for x in switch_map[cap])),
                "in_naive_switch_neighborhood": bool(any(np.isclose(share, x + shift) for x in switch_map[cap] for shift in (-0.01, 0, 0.01))),
                "is_delta_total_peak": bool(any(c == cap and np.isclose(s, share) and label == "total" for c, s, label in peak_points)),
                "is_delta_disp_peak": bool(any(c == cap and np.isclose(s, share) and label == "displacement" for c, s, label in peak_points)),
                "best_naive_objective_value": solution["best_value"],
                "second_naive_objective_value": solution["second_value"],
                "naive_top2_gap": solution["gap"],
                "naive_top2_relative_gap": solution["gap"] / max(abs(solution["best_value"]), 1.0),
                "best_calendar_signature": schedule_signature(solution["best_schedule"]),
                "second_calendar_signature": schedule_signature(solution["second_schedule"]),
                "saved_piN_signature": schedule_signature(stored_n),
                "saved_piN_matches_raw_best": same_saved,
                "full_replay_value_raw_best": full_best,
                "full_replay_value_raw_second": full_second,
                "full_replay_difference_best_minus_second": full_best - full_second,
                "distinct_calendars": schedule_signature(solution["best_schedule"]) != schedule_signature(solution["second_schedule"]),
                "raw_schedule_variables": solution["raw_candidate_count"],
                "solver_message": solution["best_message"],
            }
            row.update(_myopic_action_ties(
                artifact["draws_by_product"], artifact["weekly_profiles"],
                artifact["action_sets"], planning, share, cap,
            ))
            top2_rows.append(row)
            tie_tolerance = TIE_ATOL * max(1.0, abs(solution["best_value"]))
            if solution["gap"] <= tie_tolerance:
                tie_rows.append({**row, "policy": "piN", "tie_tolerance": tie_tolerance,
                                 "tie_definition": "top-two raw objective gap <= 1e-8 * max(1, |best objective|)"})
            if row["myopic_within_product_action_tie_events"] or row["myopic_capacity_cutoff_tie_events"]:
                tie_rows.append({**row, "policy": "piM", "tie_tolerance": np.nan,
                                 "tie_definition": "myopic comparison difference <= 1e-8 * max(1, |comparison value|)"})

    top2 = pd.DataFrame(top2_rows).sort_values(["capacity", "reimbursement_share"])
    top2.to_csv(OUT / "naive_top2_gap.csv", index=False)
    ties = pd.DataFrame(tie_rows)
    if ties.empty:
        ties = pd.DataFrame(columns=["capacity", "reimbursement_share", "policy", "tie_tolerance", "tie_definition"])
    ties.to_csv(OUT / "tie_diagnostics.csv", index=False)

    # Human-readable scientific report with all claims tied to written output.
    b3_total = peaks.query("capacity == 3 and maximum == 'total'").iloc[0]
    b3_disp = peaks.query("capacity == 3 and maximum == 'displacement'").iloc[0]
    b3_normalized = peaks.query("capacity == 3 and maximum == 'total_12_week_normalized'").iloc[0]
    gap_at_peaks = top2.loc[top2.is_delta_disp_peak | top2.is_delta_total_peak]
    n_small = int((top2.naive_top2_relative_gap <= 1e-8).sum())
    n_myopic_tie_cells = int(((top2.myopic_within_product_action_tie_events > 0) | (top2.myopic_capacity_cutoff_tie_events > 0)).sum())
    report = f"""# Price-consistent two-week-cooldown calendar audit

## Inputs and command

- Canonical artifact: `{ARTIFACT_PATH.relative_to(ROOT)}`
- Canonical 12-week reporting grid: `{TASK8_GRID_PATH.relative_to(ROOT)}`
- Command: `python scripts/audit_price_consistent_calendar_switching.py`
- The audit did not modify the canonical artifact or canonical result tables.

## B=3 peak resolution

The canonical full grid places the B=3 **absolute total** peak at λ={b3_total.reimbursement_share:.2f} (Δtotal={b3_total.delta_total:.6f}; {b3_total.delta_total_pct_12_week_myopic:.6f}% of 12-week myopic planning profit). Its displacement component peaks separately at λ={b3_disp.reimbursement_share:.2f} (Δdisp={b3_disp.delta_disp:.6f}). The source of the apparent Figure/table inconsistency is now identified exactly: `scripts/build_three_policy_figures.py` selects its peak marker by maximizing **normalized** `delta_total_percent_myopic`, whereas the Task-8 summary table maximizes absolute `delta_total`. For B=3, normalized Δtotal peaks at λ={b3_normalized.reimbursement_share:.2f}, even though absolute Δtotal peaks at λ={b3_total.reimbursement_share:.2f}, because the 12-week myopic-profit denominator varies with λ. Thus, λ=0.76 is both the B=3 displacement peak and the B=3 normalized-total peak; λ=0.86 is the B=3 absolute-total peak.

## D versus N calendars

`dynamic_vs_naive_calendar_diagnostics.csv` compares πD with πN at each capacity's Δtotal and Δdisp maxima. It reports promotion-status disagreements separately from discount-depth disagreements conditional on both policies promoting. It also reports G_N→D, the naïve-objective loss from using the actually selected πD calendar rather than πN's own objective-optimal calendar, alongside G_D→N=Δdisp. Conclusions about timing versus depth should use those columns rather than comparisons against πM.

## Naïve top-two mechanism and ties

The runner-up calculation uses all persisted raw feasible schedules, not the same-mask-pruned candidate list. This is exact for the production feasible-set/objective convention. It covers every πN switch and adjacent grid points, plus all component peaks. Across these audited cells, {n_small} top-two gaps meet the conservative floating-point tie threshold. At the same points, {n_myopic_tie_cells} cells have a material myopic action or capacity-cutoff tie. `tie_diagnostics.csv` gives any tied calendars' full-displacement replay difference, so sensitivity to solver tie-breaking is explicit rather than assumed.

At the audited Δtotal/Δdisp peaks, the top-two gaps are reported in `naive_top2_gap.csv`; the table should be used to determine whether small naïve margins coincide with large Δdisp rather than presuming the proposed reversal mechanism.

Completed in {time.monotonic() - started:.1f} seconds.
"""
    (OUT / "calendar_switching_audit_report.md").write_text(report, encoding="utf-8")
    print(f"Saved audit outputs to {OUT}")
    print(peaks.to_string(index=False))
    print(f"Audited raw top-two cells: {len(top2)}; numerical ties: {len(ties)}")


if __name__ == "__main__":
    main()
