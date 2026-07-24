# Repository cleanup report

## Completed

- Reduced the active workflow from 18 exploratory notebooks to 7 consecutively numbered notebooks.
- Merged the policy-regime interpretation and fine-grid boundary validation into one final results notebook.
- Removed the unmatched support-sensitivity branch from the active results pipeline.
- Moved superseded DRO, simulation, support-reliability, and early washout notebooks to clearly labelled archive folders.
- Replaced version-suffixed active names with canonical artifact, table, and figure paths.
- Moved reusable policy logic into `src/price_of_extrapolation/` and retained compatibility shims for historical pickle files.
- Separated stable processed data, fitted artifacts, regenerable caches, final results, and archived outputs.
- Added project metadata, minimal dependencies, citation metadata, repository checks, unit tests, cache cleanup, checksums, and artifact documentation.

## Validation performed

- Every active notebook code cell passed Python syntax parsing.
- The active notebook numbering and naming check passed.
- Lightweight policy invariants and support-table compatibility tests passed.
- Existing paper-facing figures and tables were preserved under `results/final/`.

## Not executed in this environment

The entire raw-data pipeline was not rerun because the review bundle intentionally omitted the Dominick's raw data and several large stable processed files. The included binary artifacts also require the declared PyArrow environment. After migration, run Notebooks 01–07 from fresh kernels in the local project environment and commit the executed outputs only after all assertions pass.
