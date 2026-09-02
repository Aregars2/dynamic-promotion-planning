"""Canonical paper reproduction entry point.

Use ``--verify-only`` for the inexpensive existing-artifact check. A full run
executes Notebooks 01--08 in order in fresh kernels and is intentionally
expensive.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_sample_construction.ipynb",
    "02_demand_estimation.ipynb",
    "03_descriptive_analysis.ipynb",
    "04_behavioral_calibration.ipynb",
    "05_product_calibration_and_action_support.ipynb",
    "06_policy_optimization.ipynb",
    "07_policy_results.ipynb",
    "08_robustness_and_uncertainty.ipynb",
)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce or verify canonical paper outputs.")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing outputs without recomputation.")
    args = parser.parse_args()
    if args.verify_only:
        run([sys.executable, "scripts/verify_paper_outputs.py"])
        return

    raw = ROOT / "data" / "raw"
    if not raw.is_dir() or not any(raw.iterdir()):
        raise FileNotFoundError("A clean reproduction requires the undistributed source files in data/raw/.")
    started = time.monotonic()
    for notebook in NOTEBOOKS:
        stage = time.monotonic()
        print(f"Starting {notebook}", flush=True)
        command = [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1",
        ]
        kernel = os.environ.get("PAPER_REPRO_KERNEL")
        if kernel:
            command.append(f"--ExecutePreprocessor.kernel_name={kernel}")
        run(command + [str(Path("notebooks") / notebook)])
        print(f"Completed {notebook} in {time.monotonic() - stage:.1f}s", flush=True)
    run([sys.executable, "scripts/verify_paper_outputs.py"])
    print(f"Completed canonical 01--08 reproduction in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
