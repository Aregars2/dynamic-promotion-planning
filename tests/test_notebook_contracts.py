from __future__ import annotations

import ast
from pathlib import Path

import nbformat


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_active_notebooks_parse_and_contain_no_function_definitions() -> None:
    notebook_dir = _project_root() / "notebooks"
    # Current ordered production workflow.  The retired depth-only notebook is
    # intentionally absent; Notebook 08 is the final robustness/reporting step.
    expected = [
        "01_sample_construction.ipynb",
        "02_demand_estimation.ipynb",
        "03_descriptive_analysis.ipynb",
        "04_behavioral_calibration.ipynb",
        "05_product_calibration_and_action_support.ipynb",
        "06_policy_optimization.ipynb",
        "07_policy_results.ipynb",
        "08_robustness_and_uncertainty.ipynb",
    ]
    notebooks = [notebook_dir / name for name in expected]
    assert sorted(path.name for path in notebook_dir.glob("*.ipynb")) == expected
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source)
            assert not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for node in ast.walk(tree)
            )


def test_eight_notebook_workflow_uses_versioned_eb_producers_and_consumers() -> None:
    """Guard the ordered production chain without rerunning costly estimation."""
    root = _project_root()

    def source(name: str) -> str:
        notebook = nbformat.read(root / "notebooks" / name, as_version=4)
        return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    notebook_04 = source("04_behavioral_calibration.ipynb")
    notebook_05 = source("05_product_calibration_and_action_support.ipynb")
    notebook_06 = source("06_policy_optimization.ipynb")
    notebook_07 = source("07_policy_results.ipynb")
    notebook_08 = source("08_robustness_and_uncertainty.ipynb")

    assert "EB_CALIBRATION_DIR" in notebook_04
    assert "pooled_behavioral_draws.pkl" in notebook_04
    assert "empirical_bayes" in notebook_05
    assert "build_eb_descriptive_figure.py" in notebook_05
    assert "empirical_bayes" in notebook_06
    assert "product_behavioral_draws.pkl" in notebook_06
    assert "policy_optimization.pkl" in notebook_06
    assert "task8_three_policy_reporting_grid.csv" in notebook_07
    assert "run_task7_robustness.py" in notebook_08
    assert "run_task8_reporting_checks.py" in notebook_08
    assert "build_three_policy_figures.py" in notebook_08

    for script_name in [
        "build_three_policy_figures.py",
        "run_final_robustness_reporting.py",
        "run_alternative_ppml_robustness.py",
        "run_task7_robustness.py",
        "run_task8_reporting_checks.py",
    ]:
        text = (root / "scripts" / script_name).read_text(encoding="utf-8")
        assert "empirical_bayes" in text, script_name
        assert "artifacts/policy/policy_optimization.pkl" not in text, script_name


def test_notebook_06_full_grid_call_passes_all_required_inputs() -> None:
    path = _project_root() / "notebooks" / "06_policy_optimization.ipynb"
    notebook = nbformat.read(path, as_version=4)
    matching_calls = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if name == "load_or_build_schedule_system":
                matching_calls.append(node)

    assert len(matching_calls) >= 2
    required = {
        "cache_path",
        "planning",
        "alpha_grid",
        "draws_by_product",
        "weekly_profiles",
        "action_sets",
    }
    for call in matching_calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert required.issubset(keywords)


def _assignment_value(notebook_path: Path, variable_name: str):
    notebook = nbformat.read(notebook_path, as_version=4)
    matches = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    matches.append(node.value)
    return matches


def test_notebooks_02_and_03_take_analysis_settings_from_typed_config() -> None:
    root = _project_root() / "notebooks"
    tracked = {
        "02_demand_estimation.ipynb": [
            "N_PRODUCTS",
            "TRAIN_SHARE",
            "CALIBRATION_SHARE",
            "REFIT_EVERY_N_WEEKS",
            "DISCOUNT_DEPTH_IS_PERCENT",
        ],
        "03_descriptive_analysis.ipynb": [
            "TRAIN_SHARE",
            "CALIBRATION_SHARE",
            "DEPTH_CLUSTER_WIDTH",
            "MIN_SUPPORT_OBSERVATIONS",
            "MIN_SUPPORT_PANELS",
            "MAX_POSITIVE_DEPTHS",
            "PRE_EVENT_WEEKS",
            "POST_EVENT_WEEKS",
        ],
    }
    for notebook_name, names in tracked.items():
        notebook_path = root / notebook_name
        for name in names:
            values = _assignment_value(notebook_path, name)
            assert len(values) == 1, f"{notebook_name}: expected one assignment for {name}"
            assert isinstance(values[0], ast.Attribute), (
                f"{notebook_name}: {name} must be assigned from the typed config, "
                "not a duplicated literal."
            )
