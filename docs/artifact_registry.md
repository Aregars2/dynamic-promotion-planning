# Artifact registry

| Canonical artifact | Producer | Consumers | Status |
|---|---|---|---|
| `data/processed/paper_selected_sample.parquet` | Notebook 03 | Notebooks 06 and paper tables | Stable processed sample |
| `artifacts/demand/demand_predictions.pkl` | Notebook 02 | Notebooks 03–06 | Fitted prediction artifact |
| `artifacts/demand/demand_prediction_context.pkl` | Notebook 02 | Notebook 03 | Prediction metadata |
| `artifacts/calibration/empirical_bayes/pooled_behavioral_draws.pkl` | Notebook 04 | Notebook 05 | Pooled calibration bridge |
| `artifacts/calibration/empirical_bayes/product_behavioral_draws.pkl` | Notebook 05 | Notebook 06 and Figure 1 | Product-level behavioral draws |
| `artifacts/calibration/empirical_bayes/product_calibration.pkl` | Notebook 05 | Documentation and extensions | Compact calibration metadata |
| `artifacts/calibration/empirical_bayes/supported_actions.pkl` | Notebook 05 | Notebook 06 | Product-specific feasible action sets |
| `results/empirical_bayes/tables/supported_action_clusters.csv` | Notebook 05 | Notebook 06 and paper tables | Auditable support counts and representative actions |
| `artifacts/policy/empirical_bayes_price_consistent/policy_optimization.pkl` | Notebook 06 | Notebooks 07--08 | Frozen final policy grid and decompositions |
| `results/empirical_bayes_price_consistent/tables/policy_results.csv` | Notebook 06 | Notebooks 07--08 | Human-readable main-policy grid |
| `results/empirical_bayes_price_consistent/tables/candidate_pruning_exactness.csv` | Notebook 06 | Notebook 07 | Exact candidate-pruning validation |
| `results/empirical_bayes_price_consistent/{tables,robustness,figures}/` | Notebook 08 | Paper outputs | Canonical reporting and robustness outputs |
| `artifacts/cache/policy/empirical_bayes_price_consistent/*.pkl` | Notebook 06 | Same notebook run | Regenerable; never a scientific source of truth |

The compatibility modules in `src/corrected_promotion_analysis.py` and `src/tiny_paper_pipeline_v3.py` exist only because historical pickle files encode their original Python module names. New code should import from `dynamic_promotion_planning`.
