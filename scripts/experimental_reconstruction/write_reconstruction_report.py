"""Write a concise, outcome-free report from price-side reconstruction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "experimental_reconstruction" / "reconstruction_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir

    manifest = pd.read_csv(root / "reconstruction_manifest.csv")
    study1_window = pd.read_csv(root / "study1_window.csv").iloc[0]
    study1_assignments = pd.read_csv(root / "study1_store_assignments.csv")
    study2_window = pd.read_csv(root / "study2_window.csv").iloc[0]
    study2_assignments = pd.read_csv(root / "study2_store_assignments.csv")
    leave_one_out = pd.read_csv(root / "study2_assignment_robustness.csv")
    hyper = pd.read_csv(root / "rte_hyper_candidate_episodes.csv")

    freeze_status = manifest.freeze_status.iloc[0]
    blocker = manifest.blocking_condition.iloc[0]
    base_commit = manifest.base_protocol_commit.iloc[0]
    report = (
        "# Experimental reconstruction report\n\n"
        "## Scope and outcome firewall\n\n"
        "This report covers only price-side reconstruction. The pipeline used the 28 public movement archives as the evidence universe and did not load `MOVE`, `QTY`, `PROFIT`, or `PROFIT_HEX`. No published sales or profit effects were used.\n\n"
        "## Study 1: RTE cereal\n\n"
        f"The strongest price-only 16-week candidate is weeks {int(study1_window.candidate_start_week)}--{int(study1_window.candidate_end_week)}. It is **not** a frozen experimental window. The candidate assignment table contains {len(study1_assignments)} stores, all with `assignment = unresolved`. Its markdown-depth file is a price-side diagnostic, not an IV estimate.\n\n"
        "## Study 2: store-wide pricing regime\n\n"
        f"The strongest cross-category price-only candidate is weeks {int(study2_window.candidate_start_week)}--{int(study2_window.candidate_end_week)}, using evidence from all 28 archives. The table contains {len(study2_assignments)} stores, all with `frame_status = unresolved` and `assignment = unresolved`; no published 29/29/28 count completion was used. The leave-one-category-out diagnostic has {len(leave_one_out)} rows, one for each archive. Its candidate start weeks range from {int(leave_one_out.price_only_candidate_start_week.min())} to {int(leave_one_out.price_only_candidate_start_week.max())}; this is a reconstruction sensitivity diagnostic, not a final frame decision.\n\n"
        "## Study 3: RTE Hyper Hi-Lo\n\n"
        f"The output contains {len(hyper)} price-side candidate Bonus Buy UPC-week records. Each is labeled `not_randomized`: manager selection of individual UPC episodes is not an experimental treatment. Hyper assignment remains unresolved because the Study 2 frame is not frozen.\n\n"
        "## Freeze decision\n\n"
        f"**Status: {freeze_status}.** The base protocol commit is `{base_commit}`. The reconstruction cannot be frozen because {blocker} No causal outcome analysis, policy calibration, or experimental model integration may proceed from these candidate files.\n"
    )
    (root / "reconstruction_report.md").write_text(report, encoding="utf-8")
    print(root / "reconstruction_report.md")


if __name__ == "__main__":
    main()
