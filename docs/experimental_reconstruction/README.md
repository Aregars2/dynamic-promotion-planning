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

## Study 2 evidence universe versus experimental frame

All 28 publicly available movement-category archives are retained as the
evidence universe for the Study 2 reconstruction. This does **not** imply that
every observed category or store belonged to the historical experiment.

The historical Study 2 experimental frame is a separate object to be
reconstructed from published design information and price-side evidence only:

- its category membership must identify the historical 26-category frame;
- its store membership must identify the historical 86-store frame;
- categories and stores outside that frame remain available for audit,
  falsification, and negative-control analyses where informative;
- no outside category or store may receive a Study 2 treatment label merely
  because it appears in a public movement archive.

The frozen frame must record each category and store as `in_frame`,
`out_of_frame`, or `unresolved`, with the supporting source/evidence and any
dependence on published treatment counts. Assignment starts only after this
membership table has been frozen.

`unresolved` is an admissible final reconstruction status. Neither category nor
store membership may be forced simply to attain the published 26-category or
86-store totals. Likewise, EDLP/Control/Hi-Lo labels may not be completed to
the published 29/29/28 counts unless the row is explicitly marked
`count_implied` and reported separately from independently reconstructed
labels.

Before the reconstruction freeze, no code may load `MOVE`, `QTY`, `PROFIT`,
`PROFIT_HEX`, or published sales/profit treatment-effect estimates. Attempts to
use those fields must fail rather than silently proceeding.

`scripts/experimental_reconstruction/phase0_data_audit.py` implements the
first step. Its allow-list excludes `MOVE`, `QTY`, `PROFIT`, and `PROFIT_HEX`.
Its generated files are placed in `results/experimental_reconstruction/phase0/`
and remain untracked like other reproducible outputs.

Historical design facts, their sources, and page references belong in
`reconstruction_sources.md`. No published outcome estimate may be used to
select an assignment, window, threshold, or candidate Hyper episode.
