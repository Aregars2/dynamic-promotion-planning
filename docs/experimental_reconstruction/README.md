# Experimental reconstruction workflow

This directory documents a separate, outcome-blind reconstruction of the
historical Dominick's pricing experiments. It does not alter the empirical-Bayes
price-consistent promotion-planning workflow.

The workflow is staged deliberately:

1. inventory raw files using only reconstruction-permitted fields;
2. document historical design evidence and freeze assignment/window rules;
3. reconstruct treatments without loading sales, quantity, revenue, or profit;
4. hash the frozen reconstruction;
5. conduct causal analysis only after that freeze.

`scripts/experimental_reconstruction/phase0_data_audit.py` implements the
first step. Its allow-list excludes `MOVE`, `QTY`, `PROFIT`, and `PROFIT_HEX`.
Its generated files are placed in `results/experimental_reconstruction/phase0/`
and remain untracked like other reproducible outputs.

Historical design facts, their sources, and page references belong in
`reconstruction_sources.md`. No published outcome estimate may be used to
select an assignment, window, threshold, or candidate Hyper episode.
