# Reproducibility protocol

## Computational environment

Python 3.11 or later is required. The validated direct dependency versions are recorded in `constraints.txt`.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -c constraints.txt
python scripts/check_repository.py
pytest
```

## Raw data

The Dominick's scanner files are not redistributed. Place the source files required by notebook 01 in `data/raw/`. Third-party data remain subject to their original terms of use.

## Execution order

Run the active notebooks from fresh kernels in numerical order:

1. sample construction;
2. demand estimation;
3. descriptive analysis;
4. behavioral calibration;
5. product calibration and action support;
6. policy optimization;
7. policy results;
8. global policy-transition audit.

Notebook 06 writes the selected policy artifact consumed by notebooks 07 and 08. Manufacturer funding is evaluated as a reimbursement share \(\lambda\in[0,1]\) of the regular-price markdown. Notebook 07 presents coarse-grid results and decompositions. Notebook 08 independently rebuilds the schedule system for the fine reimbursement grid and verifies agreement at overlapping grid points.

The policy artifact uses one PPML fit before the planning episode to construct all policy-horizon forecasts from known-ahead covariates. It records three calendars at every reimbursement share and capacity: myopic, forward-looking displacement-naive, and forward-looking displacement-aware. The latter two increments form an ordered, not unique, decomposition: first forward planning under the naïve objective, then displacement-aware optimization conditional on forward planning. All calendars are replayed under the same full displacement objective.

## Required numerical assertions

The final policy run must satisfy:

- dynamic value is weakly greater than myopic value under the same feasible set;
- dynamic value is nondecreasing in weekly capacity;
- VDO equals dynamic profit minus myopic profit within numerical tolerance;
- the local grid reproduces coarse-grid values at overlapping funding parameters;
- every retained policy uses the selected demand-only economic profile and terminal washout.

## Regular-price robustness

The main specification defines regular price as the rolling 90th percentile of
observed price over the configured 13-week window. This upstream choice is
evaluated by rebuilding the full workflow at the 80th, 90th, and 95th
percentiles, as listed in `config/analysis.toml` under
`[sensitivity.regular_price]`. Each scenario must use an isolated artifact
directory; scenario outputs must never overwrite the main specification.

The resulting appendix table must report, for each quantile, the promotion
share, selected products and panels, product-specific supported discount
actions, and headline VDO (including VDO percent) at the main funding and
capacity specification.

## Artifact policy

Stable processed datasets, fitted objects, caches, and publication outputs are separated according to `docs/artifact_registry.md`. Cache contents are not scientific source data and can be removed with:

```bash
python scripts/clean_cache.py
python scripts/clean_cache.py --apply
```

## Interpretation

The empirical output is a model-based counterfactual policy comparison. It does not identify the causal effect of implementing a promotion calendar. Behavioral draws represent calibrated uncertainty and should not be described as frequentist confidence intervals unless a separate inferential procedure is supplied.
