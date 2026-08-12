# Changelog

## 0.3.0 - 2026-07-28

- Renamed the project and canonical package from `price_of_extrapolation` to
  `dynamic_promotion_planning`.
- Updated active notebook, script, test, documentation, distribution, and repository
  identifiers.
- Retained the historical package namespace as a compatibility-only wrapper for
  existing pickle artifacts.
- Added tests preventing active code from reverting to the retired namespace.

## 0.2.1 - 2026-07-27

- Added an independent brute-force reference implementation for tiny policy problems.
- Added decision-critical tests for replay, cooldown, MILP/pruning exactness,
  myopic simulation, decompositions, washout, ties, solver failures, and cache inputs.
- Added optional empirical-artifact integration tests.
- Added a forecast-information audit that distinguishes common-origin forecasts from
  rolling conditional-path profiles.
- Repaired the fresh-kernel full-grid schedule-builder call in Notebook 06.
- Saved forecast-information classification in the policy artifact.

## 0.2.0 — 2026-07-24

- reorganized the empirical workflow into eight consecutively numbered notebooks;
- separated policy-result presentation from local boundary refinement;
- moved reusable notebook implementation into package modules;
- limited active notebook code cells to 80 lines and removed notebook-level definitions;
- introduced typed TOML configuration and canonical artifact paths;
- replaced informal or overstated notebook wording;
- added policy invariants, action-support tests, artifact checksums, CI, licensing, and reproducibility documentation;
- retained legacy import shims only for historical pickle compatibility.

## 0.1.0

Exploratory empirical and robustness workflow before the scientific-repository refactor.
