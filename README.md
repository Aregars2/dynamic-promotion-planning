# Dynamic Promotion Planning under Demand Displacement

This repository implements an empirical study of when forward-looking retail-promotion planning improves on a myopic weekly rule when promotions shift demand across time. The empirical application uses Dominick's cereal scanner data.

The selected empirical specification combines:

- forward-rolling PPML demand predictions;
- reduced-form product-level promotion lift and post-promotion displacement;
- product-specific empirically supported discount actions;
- category-level weekly promotion-capacity constraints;
- a 12-week decision horizon followed by a no-promotion terminal washout;
- dynamic and myopic policies evaluated under the same behavioral draws and economic inputs.

The central empirical pattern is not uniform dominance by the dynamic policy. Model-implied gains are usually modest and become larger near **asynchronous discrete policy transitions**, where the dynamic and myopic planners change promotion calendars at different funding values.

## Scientific scope

The analysis is a model-based counterfactual policy evaluation. It does not identify the causal effect of implementing the recommended calendars. Product and week decompositions are additive accounting identities under the fitted model. Behavioral draws summarize calibrated uncertainty and are not interpreted as frequentist confidence intervals.

Manufacturer funding is parameterized as a reimbursement share \(\lambda\in[0,1]\) of the regular-price markdown: the per-unit reimbursement is \(\lambda p_i^0 d_{it}\). Thus \(\lambda=0\) is no funding and \(\lambda=1\) is full markdown reimbursement.

The policy comparison uses common-origin, ex-ante PPML forecasts with known-ahead policy covariates and a fresh-start state. It reports a sequential decomposition from myopic planning, to forward planning under a displacement-naive objective, to displacement-aware forward planning; all selected calendars are replayed under the same full displacement model.

A controlled mechanism simulation is planned as a thesis extension. It will isolate demand displacement, terminal truncation, weak action support, and policy-transition recovery under a known data-generating process.

## Repository structure

```text
config/                       canonical analysis configuration
notebooks/                    active empirical workflow
src/dynamic_promotion_planning/   reusable implementation
artifacts/                    compact fitted and policy objects
artifacts/cache/              regenerable schedule caches
data/processed/               stable processed samples
results/final/                publication tables and figures
archive/                      superseded analyses and compatibility material
scripts/                      repository checks and cleanup utilities
tests/                        deterministic unit and invariant tests
```

## Active workflow

Run the notebooks in order from fresh kernels:

1. `01_sample_construction.ipynb`
2. `02_demand_estimation.ipynb`
3. `03_descriptive_analysis.ipynb`
4. `04_behavioral_calibration.ipynb`
5. `05_product_calibration_and_action_support.ipynb`
6. `06_policy_optimization.ipynb`
7. `07_policy_results.ipynb`
8. `08_policy-transition_audit.ipynb` — global 0.01 policy-transition audit across all reported capacities

The notebooks state the empirical design, call reusable package functions, display diagnostics, and export canonical artifacts. Function definitions and reusable computational routines belong in `src/dynamic_promotion_planning/`.

## Reproduction

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -c constraints.txt
```

On Windows, activation is optional; commands may be run directly with:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\check_repository.py
```

Run the repository checks before and after executing the pipeline:

```bash
python scripts/check_repository.py
python scripts/verify_artifact_checksums.py
pytest
```

Raw Dominick's files are not distributed. Full reproduction from raw scanner data requires the original files in `data/raw/`. The complete execution and interpretation protocol is documented in `docs/reproducibility.md`; refactor checks are summarized in `docs/refactor_validation.md`.

## Configuration

`config/analysis.toml` is the canonical source for sample restrictions, demand-selection thresholds, behavioral settings, action-support rules, policy grids, a fixed conservative terminal washout, and the global policy-transition audit grid.

## Data and artifact policy

- Stable processed samples belong in `data/processed/`.
- Compact fitted objects belong in `artifacts/{demand,calibration,policy}/`.
- Regenerable schedule systems belong in `artifacts/cache/` and are ignored by Git.
- Publication-facing outputs belong in `results/final/`.
- Superseded experiments belong in `archive/` or Git history.
- Version suffixes such as `_v2`, `_revised`, and `_final` are not used for active artifacts.

## Compatibility

The canonical Python namespace is `dynamic_promotion_planning`. The historical
`price_of_extrapolation` namespace remains only as a thin compatibility layer because
existing pickle artifacts encode their original module paths. Active notebooks,
scripts, tests, and new code must import from `dynamic_promotion_planning`.

The top-level modules `corrected_promotion_analysis.py` and
`tiny_paper_pipeline_v3.py` are also retained solely for historical pickle
compatibility. They are not part of the active public API.

## License

Python source code and code cells are licensed under the MIT License.

Paper text, documentation, notebook narrative, figures, and tables are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), unless otherwise stated.

Third-party data are not covered by these licenses. Dominick's scanner data are not redistributed and remain subject to their original terms.

## Development

Repository conventions and pre-commit checks are documented in `CONTRIBUTING.md`.


## Decision-critical verification

The test suite includes an independent tiny-problem reference optimizer that does
not import the production policy implementation. It checks direct schedule replay,
cooldown feasibility, exhaustive-search versus MILP solutions, pruning exactness,
myopic replay, decomposition identities, washout, ties, solver failures, and cache
fingerprints.

The weekly-profile information set can be audited with:

```powershell
.\.venv\Scripts\python.exe scripts\audit_forecast_information.py
```

The command distinguishes a common-origin ex-ante forecast from a rolling
conditional-path profile. Full details are in `docs/test_protocol.md`.
