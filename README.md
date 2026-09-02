# Dynamic Promotion Planning under Demand Displacement

This repository contains the code and reproducible workflow for a model-based
study of retail promotion planning using Dominick's cereal scanner data. It
compares a weekly myopic policy, a forward planner that ignores modeled
post-promotion displacement from candidate promotions, and a dynamic forward
planner that accounts for those modeled future demand effects. The policies use
common-origin PPML profiles and common behavioral replay.

The results are counterfactual policy comparisons under the estimated demand
model. They should not be interpreted as causal estimates of implementing the
selected promotion calendars.

The current GitHub repository is `dynamic-promotion-planning`, matching the
scientific workflow documented here.

## Setup and reproduction

Python 3.11+ is required. Raw data are not redistributed; place them in `data/raw/` for a clean reproduction.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -c constraints.txt
python scripts/reproduce_paper.py --verify-only
```

The verification path is inexpensive. `python scripts/reproduce_paper.py` is expensive: it refits PPML, calibrates the behavioral draws, solves the policy grid, and runs robustness analyses.

## Canonical notebooks

Run the notebooks in the following order, using a fresh kernel for each:

1. `01_sample_construction`
2. `02_demand_estimation`
3. `03_descriptive_analysis`
4. `04_behavioral_calibration`
5. `05_product_calibration_and_action_support`
6. `06_policy_optimization`
7. `07_policy_results`
8. `08_robustness_and_uncertainty`

Notebook 07 contains schedule diagnostics and validation material that is not a current manuscript result. Notebook 08 creates price-consistent reporting, robustness, uncertainty, Figure 2, and Tables 3--5.

## Paper outputs

| Object | Producer | Canonical output |
| --- | --- | --- |
| Table 1 | Notebook 03 | `results/empirical_bayes/tables/paper_product_characteristics.{csv,tex}` |
| Table 2 | Notebook 02 | `results/final/tables/final_demand_model_comparison.csv`; `paper/tables/table2_demand_model_performance.tex` |
| Figure 1 | Notebook 05 | `results/empirical_bayes/figures/paper_promotion_depths_and_model_implied_post_effects.{png,pdf}` |
| Figure 2 and Tables 3--5 | Notebook 08 | `results/empirical_bayes_price_consistent/`; `paper/tables/` |

Run `python scripts/verify_paper_outputs.py` after reproduction. Generated tables can be included through `\input{paper/tables/<file>.tex}` if an authoritative manuscript source is supplied.

## Repository layout

- `config/`: canonical analysis settings.
- `src/dynamic_promotion_planning/`: implementation.
- `data/processed/`: stable analysis inputs; `data/raw/`: local raw data.
- `artifacts/`: generated fitted models and policy objects.
- `artifacts/cache/`: regenerable schedule caches.
- `results/`: generated analysis outputs.
- `paper/tables/`: deterministic LaTeX tables.
- `scripts/audits/`: retained price-consistent switching and calibration-safeguard audits.
- `tests/`: deterministic and integration checks.

## Checks

```bash
python -m pytest
python scripts/check_repository.py
python scripts/verify_paper_outputs.py
```

See `docs/reproducibility.md`, `docs/artifact_registry.md`, `docs/paper_output_coverage.md`, `docs/test_protocol.md`, and `docs/scope_and_limitations.md` for detailed contracts.
