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
    "03_descriptive_analysis.ipynb",
    "04_behavioral_calibration.ipynb",
    "05_product_calibration_and_action_support.ipynb",
    "06_policy_optimization.ipynb",
    "07_policy_results.ipynb",
    "08_boundary_refinement.ipynb",
]
MAX_CODE_CELL_LINES = 80
MAX_ACTIVE_FUNCTION_LINES = 250
PROHIBITED_HEADINGS = {
    "Save human-readable summaries",
    "Create paper-ready tables",
    "Notes for the paper",
    "Interpretation checklist",
    "Interpretation guardrails",
    "validates the mechanism",
    "exact counterfactual prediction",
}

errors: list[str] = []
actual = sorted(path.name for path in ACTIVE.glob("*.ipynb"))
if actual != EXPECTED:
    errors.append(f"Active notebook set differs from expected:\n{actual}")

for path in ACTIVE.glob("*.ipynb"):
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            line_count = len(cell.source.splitlines())
            if line_count > MAX_CODE_CELL_LINES:
                errors.append(
                    f"{path.name}, cell {index}: {line_count} code lines "
                    f"(maximum {MAX_CODE_CELL_LINES})"
                )
            try:
                tree = ast.parse(cell.source)
            except SyntaxError as exc:
                errors.append(f"{path.name}, cell {index}: {exc}")
                continue
            definitions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                )
            ]
            if definitions:
                errors.append(
                    f"{path.name}, cell {index}: reusable definitions belong in src/: "
                    f"{definitions}"
                )
        elif cell.cell_type == "markdown":
            for phrase in PROHIBITED_HEADINGS:
                if phrase.lower() in cell.source.lower():
                    errors.append(
                        f"{path.name}, cell {index}: informal or overstated wording "
                        f"{phrase!r}"
                    )

    if re.search(
        r"(?:^|[_-])(?:v\d+|revised|refined)(?:[_-]|$)",
        path.stem,
        re.I,
    ):
        errors.append(f"Version suffix in active notebook name: {path.name}")


ACTIVE_PACKAGE = ROOT / "src" / "dynamic_promotion_planning"
SOURCE_EXCLUSIONS = {"legacy_pipeline.py"}
for path in ACTIVE_PACKAGE.glob("*.py"):
    if path.name in SOURCE_EXCLUSIONS:
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        errors.append(f"{path.relative_to(ROOT)}: {exc}")
        continue
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if node.end_lineno is None:
            continue
        line_count = node.end_lineno - node.lineno + 1
        if line_count > MAX_ACTIVE_FUNCTION_LINES:
            errors.append(
                f"{path.relative_to(ROOT)}:{node.lineno}: {node.name} spans "
                f"{line_count} lines (maximum {MAX_ACTIVE_FUNCTION_LINES})"
            )

if not (ROOT / "config" / "analysis.toml").is_file():
    errors.append("Missing canonical config/analysis.toml")
if not (ROOT / "LICENSE").is_file():
    errors.append("Missing software LICENSE")

if errors:
    print("Repository checks failed:")
    for error in errors:
        print(" -", error)
    sys.exit(1)

print(f"Repository checks passed for {len(EXPECTED)} active notebooks.")
