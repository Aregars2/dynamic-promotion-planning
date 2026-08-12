from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dynamic_promotion_planning.policy import load_pickle


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_policy_artifact():
    pytest.importorskip("pyarrow")
    path = _project_root() / "artifacts" / "policy" / "policy_optimization.pkl"
    if not path.is_file():
        pytest.skip("Canonical policy artifact is not present.")
    try:
        return load_pickle(path)
    except ModuleNotFoundError as error:
        pytest.skip(f"Artifact compatibility dependency unavailable: {error}")


@pytest.mark.integration
def test_empirical_policy_grid_invariants() -> None:
    artifact = _load_policy_artifact()
    results = artifact["policy_results"].copy()
    required = {
        "alpha",
        "capacity",
        "dynamic_profit",
        "myopic_profit",
        "vdo",
        "economic_profile_mode",
    }
    assert required.issubset(results.columns)
    assert len(results) == 152
    assert sorted(results["capacity"].unique().tolist()) == [1, 2, 3, 8]
    assert np.allclose(
        results["vdo"],
        results["dynamic_profit"] - results["myopic_profit"],
        atol=1e-6,
        rtol=0.0,
    )
    assert (results["dynamic_profit"] >= results["myopic_profit"] - 1e-6).all()

    wide = results.pivot(index="alpha", columns="capacity", values="dynamic_profit")
    assert (wide.diff(axis=1).iloc[:, 1:] >= -1e-6).all().all()


@pytest.mark.integration
def test_empirical_decompositions_reconcile() -> None:
    artifact = _load_policy_artifact()
    results = artifact["policy_results"]
    product = artifact["product_decomposition"]
    weekly = artifact["weekly_decomposition"]

    product_sum = (
        product.groupby(["alpha", "capacity"], observed=True)["vdo_contribution"]
        .sum()
        .rename("product_vdo")
        .reset_index()
    )
    weekly_sum = (
        weekly.groupby(["alpha", "capacity"], observed=True)["vdo_contribution"]
        .sum()
        .rename("weekly_vdo")
        .reset_index()
    )
    merged = (
        results[["alpha", "capacity", "vdo"]]
        .merge(product_sum, on=["alpha", "capacity"], validate="one_to_one")
        .merge(weekly_sum, on=["alpha", "capacity"], validate="one_to_one")
    )
    assert np.allclose(merged["vdo"], merged["product_vdo"], atol=1e-5)
    assert np.allclose(merged["vdo"], merged["weekly_vdo"], atol=1e-5)


@pytest.mark.integration
def test_empirical_profiles_use_fixed_price_and_cost() -> None:
    artifact = _load_policy_artifact()
    profiles = artifact["weekly_profiles"]
    assert all(
        np.allclose(np.asarray(values["price_factor"], dtype=float), 1.0)
        for values in profiles.values()
    )
    assert all(
        np.allclose(np.asarray(values["cost_factor"], dtype=float), 1.0)
        for values in profiles.values()
    )
