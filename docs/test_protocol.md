# Decision-critical test protocol

The unit-test suite now includes an independent tiny-problem implementation under
`tests/reference_policy.py`. That reference code does not import the production
policy module. It independently replays schedules, enumerates cooldown-feasible
calendars, solves tiny category problems by exhaustive search, and simulates the
myopic rule.

The main deterministic checks cover:

- exact schedule replay;
- cooldown equivalence;
- affine values in the funding parameter;
- occupancy pruning and MILP solutions against exhaustive search;
- myopic replay and independent myopic simulation;
- dynamic dominance and capacity monotonicity;
- action-capacity feasibility;
- product and week decomposition identities;
- cache-fingerprint sensitivity;
- no-action and no-displacement placebos;
- a strong-displacement delay example;
- exact ties and second-best failure handling;
- washout-state decay;
- draw-specific state propagation;
- serialization round trips;
- coarse/fine-grid overlap and boundary classification;
- notebook syntax and the Notebook 06 schedule-builder contract.

## Forecast-information audit

`dynamic_promotion_planning.forecast_audit` distinguishes three relevant states:

- `ex_ante_common_origin`: every model fit precedes the common planning origin and
  future-covariate availability has been separately verified;
- `common_origin_fit_covariates_unverified`: model timing is valid, but future
  covariates have not been established as operationally available;
- `rolling_conditional_path`: at least one prediction uses a model refitted inside
  the planning horizon.

The audit deliberately does not infer future-covariate availability from model-fit
dates. Lagged sales, future prices, or other realized-path variables require a
separate construction audit or recursive forecast design.

Run:

```text
python scripts/audit_forecast_information.py
```

Do not use `--future-covariates-verified` unless every covariate entering every
profile week has been traced to information available at the common planning origin.

## Integration tests

Tests marked `integration` inspect canonical generated artifacts when their runtime
dependencies and files are available. They validate the 152-row coarse grid,
dynamic dominance, capacity monotonicity, fixed price/cost factors, and additive
decompositions. They skip rather than fail when generated artifacts or compatibility
dependencies are absent.
