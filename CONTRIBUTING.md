# Contributing to the analysis

This repository is maintained as a scientific workflow rather than as a general-purpose software package.

## Development principles

- Put reusable computation in `src/dynamic_promotion_planning/`.
- Keep notebooks focused on empirical design, orchestration, diagnostics, and displayed results.
- Do not define functions or classes in active notebooks.
- Keep each notebook code cell below 80 lines and limited to one analytical step.
- Read analysis choices from `config/analysis.toml` rather than duplicating constants.
- Use canonical artifact paths documented in `docs/artifact_registry.md`.
- Store regenerable schedule systems only under `artifacts/cache/`.
- Do not introduce active filenames containing version suffixes such as `_v2`, `_revised`, or `_final`.
- State model-based counterfactual findings separately from causal claims.

## Before committing

```bash
python scripts/check_repository.py
pytest
```

Changes to the selected empirical specification should also rerun notebooks 06--08 from fresh kernels. Changes to upstream data or calibration logic require a complete 01--08 run.
