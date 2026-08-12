# Empirical-Bayes behavioral calibration

The post-`pre-eb-heuristic` calibration replaces the retired support-count
weight with parameter-specific bootstrap-precision empirical-Bayes weights.
For each product and each policy-scale parameter (price elasticity, promotion
lift, and displacement strength), the implementation computes the variance of
the transformed/clipped product bootstrap draws, estimates the between-product
variance by method of moments, and mixes product bootstrap parent `m` with the
same aligned pooled bootstrap parent `m`.

Persistence remains governed by the existing identifiable-decay and fallback
grid logic.  It is not assigned an EB weight.

The historic support-count weight is retained only as a diagnostic column
(`legacy_heuristic_reliability`); it does not enter any calibration or policy
calculation.  EB calibration products are versioned beneath
`artifacts/calibration/empirical_bayes`, and later policy products are
versioned beneath `artifacts/policy/empirical_bayes` and
`results/empirical_bayes`.
