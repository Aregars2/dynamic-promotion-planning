from __future__ import annotations

from pathlib import Path
import importlib
import tomllib

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_and_canonical_package_names() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    assert pyproject["project"]["name"] == "dynamic-promotion-planning"
    assert (ROOT / "src" / "dynamic_promotion_planning").is_dir()


def test_active_code_uses_only_canonical_namespace() -> None:
    active_paths = [
        *sorted((ROOT / "notebooks").glob("*.ipynb")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]

    for path in active_paths:
        if path.name == "test_namespace_migration.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "from price_of_extrapolation" not in text
        assert "import price_of_extrapolation" not in text


def test_notebook_imports_use_canonical_namespace() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert "price_of_extrapolation" not in code
        assert "dynamic_promotion_planning" in code


def test_historical_namespace_resolves_to_canonical_objects() -> None:
    canonical = importlib.import_module("dynamic_promotion_planning.policy")
    historical = importlib.import_module("price_of_extrapolation.policy")

    assert historical.PlanningSpec is canonical.PlanningSpec
    assert historical.SupportSpec is canonical.SupportSpec
    assert historical.load_pickle is canonical.load_pickle


def test_legacy_top_level_pickle_modules_remain_importable() -> None:
    corrected = importlib.import_module("corrected_promotion_analysis")
    tiny = importlib.import_module("tiny_paper_pipeline_v3")

    assert hasattr(corrected, "PlanningSpec")
    assert tiny is not None
