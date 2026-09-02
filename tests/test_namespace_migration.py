from __future__ import annotations

from pathlib import Path
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


def test_notebooks_do_not_import_retired_namespace() -> None:
    """Notebooks may delegate to scripts; they need not import the package directly."""
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert "price_of_extrapolation" not in code


def test_release_tree_excludes_historical_pickle_compatibility() -> None:
    assert not (ROOT / "src" / "price_of_extrapolation").exists()
    assert not (ROOT / "src" / "corrected_promotion_analysis.py").exists()
    assert not (ROOT / "src" / "tiny_paper_pipeline_v3.py").exists()
    assert not (ROOT / "src" / "dynamic_promotion_planning" / "legacy_pipeline.py").exists()
