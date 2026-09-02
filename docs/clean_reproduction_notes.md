# Clean-reproduction notes

This note records non-scientific reproducibility repairs made while executing
the canonical Notebook 01--08 sequence from fresh kernels.  It is not a source
of scientific results and does not alter any model, calibration, optimization,
or reporting definition.

## Repairs

1. **Notebook 02 CSV round-trip assertion.** The paper-facing model-comparison
   table is written to CSV and read back only to confirm the export. Pandas may
   load a categorical column as an Arrow string column, even when values are
   identical. The assertion therefore ignores categorical dtype representation
   while continuing to require identical values and all other relevant table
   content.
2. **Notebook 06 to Notebook 07 audit handoff.** Notebook 07 displays the
   same-mask candidate-pruning exactness audit. Notebook 06 now invokes the
   existing `scripts/verify_candidate_pruning.py` immediately after saving its
   canonical price-consistent policy artifact, writing
   `results/empirical_bayes_price_consistent/tables/candidate_pruning_exactness.csv`.
   This removes the former manual prerequisite without changing candidate
   construction or policy calculations.

## Windows execution environment

For an isolated local clean run, the execution launcher may set `IPYTHONDIR` to
a repository-local temporary directory and set `JUPYTER_PATH` to the local
kernel-spec directory when using an explicitly installed private kernel. These
settings avoid user-profile history permissions and are not inputs to any
scientific computation. The default `python -m jupyter nbconvert` path remains
the canonical entry point for ordinary use.

## Validation standard

After the notebook sequence completes, run:

```text
python scripts/verify_paper_outputs.py
python -m pytest
python scripts/check_repository.py
```

The verification script rebuilds deterministic manuscript Tables 3--5 from
their canonical reporting inputs; it does not re-estimate or re-optimize.
