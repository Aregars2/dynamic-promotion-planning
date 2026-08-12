"""Canonical repository paths and artifact names."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing ``pyproject.toml`` and ``data``."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the project root from {current}. "
        "Run from the repository or a subdirectory."
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def artifact_demand(self) -> Path:
        return self.artifacts / "demand"

    @property
    def artifact_calibration(self) -> Path:
        return self.artifacts / "calibration"

    @property
    def artifact_policy(self) -> Path:
        return self.artifacts / "policy"

    @property
    def artifact_cache(self) -> Path:
        return self.artifacts / "cache"

    @property
    def result_tables(self) -> Path:
        return self.root / "results" / "final" / "tables"

    @property
    def result_figures(self) -> Path:
        return self.root / "results" / "final" / "figures"

    @property
    def result_models(self) -> Path:
        return self.root / "results" / "models"

    def create_output_directories(self) -> None:
        for path in [
            self.data_raw,
            self.data_processed,
            self.artifact_demand,
            self.artifact_calibration,
            self.artifact_policy,
            self.artifact_cache,
            self.result_tables,
            self.result_figures,
            self.result_models,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def project_paths(start: Path | None = None) -> ProjectPaths:
    paths = ProjectPaths(find_project_root(start))
    paths.create_output_directories()
    return paths
