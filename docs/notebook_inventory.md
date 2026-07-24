# Notebook inventory

## Active empirical pipeline

| New notebook | Source notebook(s) | Role |
|---|---|---|
| `01_sample_construction.ipynb` | `01_sample_creation.ipynb` | Scanner-data cleaning, prices, margins, and modeling sample |
| `02_demand_estimation.ipynb` | `02_demand_model_selection.ipynb` | Forward-rolling demand-model comparison and predictions |
| `03_paper_sample_and_descriptives.ipynb` | `03_paper_data_outputs_residualized_fixed.ipynb` | Frozen paper sample, descriptive tables, and Figure 1 inputs |
| `04_behavioral_calibration.ipynb` | `04_stockpiling_calibration_revised_v2.ipynb` | Pooled reduced-form promotion dynamics and behavioral uncertainty |
| `05_product_calibration_and_action_support.ipynb` | `07b_product_calibration_and_supported_actions.ipynb` | Product-level calibration, holdout checks, and supported actions |
| `06_policy_optimization.ipynb` | `10_corrected_policy_optimization_minimal.ipynb` | Frozen terminally corrected dynamic versus myopic policy grid |
| `07_results_and_boundary_validation.ipynb` | `11_policy_regimes_and_support.ipynb` + `12_boundary_validation.ipynb` | Results, decompositions, and fine-grid boundary mechanism validation |

## Archived: future methodological extensions

The notebooks in `archive/notebooks/dro/` contain the earlier support-aware DRO, persistent-parameter, contract-threshold, and simulation work. They are scientifically relevant for a later thesis extension but are not required for the current empirical paper.

## Archived: superseded empirical implementations

The notebooks in `archive/notebooks/legacy/` contain product-frontier, category-capacity, earlier washout, and pre-frozen optimization implementations. They should not be used to regenerate the paper results.

## Archived: earlier research direction

`archive/notebooks/earlier_research_direction/local_support_and_reliability.ipynb` belongs to the earlier local-counterfactual-support pricing project. It is retained for conceptual continuity but excluded from this promotion-planning pipeline.
