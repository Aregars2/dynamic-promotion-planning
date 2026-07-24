# Artifact registry

| Canonical artifact | Producer | Consumers | Status |
|---|---|---|---|
| `data/processed/paper_selected_sample.parquet` | Notebook 03 | Notebooks 06 and paper tables | Stable processed sample |
| `artifacts/demand/demand_predictions.pkl` | Notebook 02 | Notebooks 03–06 | Fitted prediction artifact |
| `artifacts/demand/demand_prediction_context.pkl` | Notebook 02 | Notebook 03 | Prediction metadata |
| `artifacts/calibration/pooled_behavioral_draws.pkl` | Notebook 04 | Notebook 05 | Pooled calibration bridge |
| `artifacts/calibration/product_behavioral_draws.pkl` | Notebook 05 | Notebook 06 | Product-level behavioral draws |
| `artifacts/calibration/product_calibration.pkl` | Notebook 05 | Documentation and extensions | Compact calibration metadata |
| `artifacts/calibration/supported_actions.pkl` | Notebook 05 | Notebook 06 | Product-specific feasible action sets |
| `results/final/tables/supported_action_clusters.csv` | Notebook 05 | Notebook 06 and paper tables | Auditable support counts and representative actions |
| `artifacts/policy/policy_optimization.pkl` | Notebook 06 | Notebook 07 | Frozen final policy grid and decompositions |
| `artifacts/policy/results_analysis.pkl` | Notebook 07 | Paper writing | Coarse and fine-grid result diagnostics |
| `artifacts/cache/policy/*.pkl` | Notebooks 06–07 | Same notebook run | Regenerable; never a scientific source of truth |

The compatibility modules in `src/corrected_promotion_analysis.py` and `src/tiny_paper_pipeline_v3.py` exist only because historical pickle files encode their original Python module names. New code should import from `price_of_extrapolation`.
