# Dynamic Promotion Planning under Demand Displacement

This repository contains the code and workflow for a study of retail promotion
planning using Dominick's cereal scanner data. It compares a weekly myopic
policy, a forward planner that ignores post-promotion displacement created by
candidate promotions, and a dynamic forward planner that accounts for those
future demand effects. All policies use the same forecast origin, baseline
demand profiles, and behavioral draws.

The results are counterfactual policy comparisons under the estimated demand
model and should not be interpreted as causal estimates of implementing the
selected promotion calendars.

## Setup and reproduction

Python 3.11+ is required. Raw data are not redistributed; place them in `data/raw/` for a clean reproduction.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -c constraints.txt
python scripts/reproduce_paper.py --verify-only
```

`python scripts/reproduce_paper.py --verify-only` verifies existing outputs
without rerunning the analysis. Running

```bash
python scripts/reproduce_paper.py
```

performs the full analysis, including PPML estimation, behavioral calibration,
policy optimization, and robustness checks.

## Analysis notebooks

Run the notebooks in the following order, using a fresh kernel for each:

1. `01_sample_construction`
2. `02_demand_estimation`
3. `03_descriptive_analysis`
4. `04_behavioral_calibration`
5. `05_product_calibration_and_action_support`
6. `06_policy_optimization`
7. `07_policy_results`
8. `08_robustness_and_uncertainty`

Notebook 07 contains schedule diagnostics and validation material that is not
used in the paper. Notebook 08 produces the final policy results, robustness
and uncertainty analyses, Figure 2, and Tables 3--5.

## Paper outputs

| Object | Producer | Output |
| --- | --- | --- |
| Table 1 | Notebook 03 | `results/empirical_bayes/tables/paper_product_characteristics.{csv,tex}` |
| Table 2 | Notebook 02 | `results/final/tables/final_demand_model_comparison.csv`; `paper/tables/table2_demand_model_performance.tex` |
| Figure 1 | Notebook 05 | `results/empirical_bayes/figures/paper_promotion_depths_and_model_implied_post_effects.{png,pdf}` |
| Figure 2 and Tables 3--5 | Notebook 08 | `results/empirical_bayes_price_consistent/`; `paper/tables/` |

After reproduction, run

```bash
python scripts/verify_paper_outputs.py
```

to verify the generated paper outputs. Generated tables can be included with
`\input{paper/tables/<file>.tex}`.

## Repository layout

- `config/`: analysis settings.
- `src/dynamic_promotion_planning/`: implementation.
- `data/processed/`: processed analysis inputs.
- `data/raw/`: local raw data.
- `artifacts/`: fitted models and policy objects.
- `artifacts/cache/`: regenerable schedule caches.
- `results/`: generated analysis outputs.
- `paper/tables/`: generated LaTeX tables.
- `scripts/audits/`: scientific and implementation checks.
- `tests/`: deterministic and integration tests.

## Checks

```bash
python -m pytest
python scripts/check_repository.py
python scripts/verify_paper_outputs.py
```

See `docs/reproducibility.md`, `docs/artifact_registry.md`,
`docs/paper_output_coverage.md`, `docs/test_protocol.md`, and
`docs/scope_and_limitations.md` for details.
