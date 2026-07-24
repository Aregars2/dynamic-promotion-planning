# Generated-data cleanup plan

The original `data/processed/` directory mixed stable data, final fitted artifacts, large caches, and outputs from superseded research branches. The cleaned repository separates these roles.

## Safe cleanup after the new pipeline has run successfully

The audit identifies **943.3 MB** of explicitly regenerable schedule and robustness caches. These can be deleted after the pre-cleanup Git snapshot and after confirming that Notebooks 01–07 run successfully.

Safe cache patterns include:

```text
data/processed/robustness_cache/
data/processed/product_schedule_artifact_*.pkl
data/processed/product_schedule_frontier_artifact.pkl
data/processed/corrected_schedule_*.pkl
data/processed/boundary_validation_schedule_system_*.pkl
data/processed/corrected_support_schedule_system_*.pkl
```

The cleaned notebooks write replacement caches to `artifacts/cache/policy/`, which is ignored by Git and can be removed with:

```bash
python scripts/clean_cache.py          # dry run
python scripts/clean_cache.py --apply  # delete
```

## Versioned duplicates

The audit identifies **1.0 MB** of explicitly version-suffixed duplicates. Do not delete these until the canonical notebooks have regenerated:

```text
artifacts/demand/demand_predictions.pkl
artifacts/calibration/product_behavioral_draws.pkl
artifacts/calibration/product_calibration.pkl
artifacts/calibration/supported_actions.pkl
artifacts/policy/policy_optimization.pkl
artifacts/policy/results_analysis.pkl
```

After that check, outputs containing `_v2`, `_revised`, or `_refined` can be removed from the active data directory. Their scientific history remains in Git and, where useful, under `archive/`.

## Methodological extensions

Approximately **154.9 MB** belongs to the earlier support-aware/DRO direction. These objects are not required for the empirical promotion-planning paper. Store them outside the active pipeline, preferably in a tagged release or an external archive, rather than in `data/processed/`.

## Stable processed data

Keep the scanner-derived Parquet files and paper sample in `data/processed/`. They are data products, not optimizer caches. Full decisions and rationales are recorded in `docs/original_artifact_audit.csv`.
