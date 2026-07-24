# Migration from the exploratory repository

1. Commit the current repository before replacing files.
2. Copy the cleaned repository contents into a new branch.
3. Preserve your local `data/raw/` directory; it is intentionally absent here.
4. Do not copy the old `data/processed/robustness_cache/` or `product_schedule_artifact_*` files.
5. Install the cleaned package and run `python scripts/check_repository.py`.
6. Run Notebooks 01–07 from fresh kernels.
7. Confirm that Notebook 07 reproduces Notebook 06 at overlapping funding values and that all dynamic-profit dominance assertions pass.
8. Commit the executed final notebooks and `results/final/`.

The complete exploratory history remains available under `archive/` and in the pre-cleanup Git commit.
