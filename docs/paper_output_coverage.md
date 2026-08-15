# Paper-output coverage and execution order

The computational companion is the ordered notebook sequence below.  Each
notebook can call tested code in `src/` or `scripts/`, but no paper-facing
output requires a manual script invocation outside this sequence.

| Paper-facing output | Producing notebook | Machine-readable output |
| --- | --- | --- |
| Sample definition, product characteristics, descriptive statistics, and promotion support | 03 — Descriptive analysis | `results/empirical_bayes/tables/paper_*.csv` |
| Figure 1: observed promotion support and model-implied post-promotion effects | 05 — Product calibration and empirical action support | `results/empirical_bayes/figures/paper_promotion_depths_and_model_implied_post_effects.{png,pdf}` and `paper_model_implied_post_promotion_effects.csv` |
| Pooled and product-level behavioral-calibration diagnostics | 04 — Behavioral calibration; 05 — Product calibration | `results/empirical_bayes/tables/pooled_*.csv`, `product_*.csv`, and `empirical_bayes_*.csv` |
| Main three-policy grid and exact decomposition | 06 — Policy optimization | `artifacts/policy/empirical_bayes/policy_optimization.pkl` and `results/empirical_bayes/tables/policy_results.csv` |
| Main policy summary and local schedule diagnostics | 07 — Policy results | `results/empirical_bayes/tables/three_policy_sequential_decomposition.csv` and related diagnostic CSVs |
| Figure 2: three-policy value contrasts by capacity | 08 — Robustness and uncertainty | `results/empirical_bayes/figures/three_policy_components_by_capacity.{png,pdf}` |
| Figure 3: calendar switches for B=2 | 08 — Robustness and uncertainty | `results/empirical_bayes/figures/three_policy_switching_rug_B2.{png,pdf}` and `three_policy_transition_table_B2.csv` |
| Reporting denominators, terminal-tail diagnostic, robustness summaries, and fixed-calendar behavioral-parameter uncertainty | 08 — Robustness and uncertainty | `results/empirical_bayes/tables/task8_*.csv` and `results/empirical_bayes/robustness/` |

The primary paper percentage denominator is 12-week myopic planning profit.
The same-horizon 48-week normalization is reported as a secondary comparison.
All policy-facing outputs use the empirical-Bayes artifacts under
`artifacts/calibration/empirical_bayes/` and
`artifacts/policy/empirical_bayes/`.

The legacy heuristic artifacts are intentionally retained outside the active
workflow for provenance.  They are not inputs to the tables or figures above.
