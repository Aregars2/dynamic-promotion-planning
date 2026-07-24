from __future__ import annotations

from pathlib import Path
import ast
import re
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "notebooks"
EXPECTED = [
    "01_sample_construction.ipynb",
    "02_demand_estimation.ipynb",
    "03_paper_sample_and_descriptives.ipynb",
    "04_behavioral_calibration.ipynb",
    "05_product_calibration_and_action_support.ipynb",
    "06_policy_optimization.ipynb",
    "07_results_and_boundary_validation.ipynb",
]

errors: list[str] = []
actual = sorted(path.name for path in ACTIVE.glob("*.ipynb"))
if actual != EXPECTED:
    errors.append(f"Active notebook set differs from expected:\n{actual}")

for path in ACTIVE.glob("*.ipynb"):
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            try:
                ast.parse(cell.source)
            except SyntaxError as exc:
                errors.append(f"{path.name}, cell {index}: {exc}")
    if re.search(r"(?:^|[_-])(?:v\d+|revised|refined)(?:[_-]|$)", path.stem, re.I):
        errors.append(f"Version suffix in active notebook name: {path.name}")

if errors:
    print("Repository checks failed:")
    for error in errors:
        print(" -", error)
    sys.exit(1)

print(f"Repository checks passed for {len(EXPECTED)} active notebooks.")
