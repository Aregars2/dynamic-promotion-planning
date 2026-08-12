# Parallel-workflow snapshot import

This branch was created from the existing `price-of-extrapolation` repository
and then populated with a point-in-time snapshot of the parallel
`dynamic_promotion_planning` workflow.

The snapshot import is intentionally not presented as a reconstruction of the
intervening parallel development history.  It records only the current source
state imported onto the provenance-root branch.

Generated data and scientific result artifacts were not copied into this Git
snapshot.  The pre-EB heuristic artifacts remain at their original workflow
paths, and `pre_eb_heuristic_artifact_hashes.sha256` records SHA-256 hashes for
the key calibration and policy outputs so that the pre-EB numerical state can
be audited independently.

The annotated `pre-eb-heuristic` tag identifies this source snapshot before
the empirical-Bayes calibration replacement.
