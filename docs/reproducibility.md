# Reproducibility protocol

Install Python 3.11+ with `python -m pip install -e ".[dev]" -c constraints.txt`.
The undistributed Dominick's source files must be placed in `data/raw/`.

The canonical scientific order is Notebooks 01--08: sample construction,
demand estimation, descriptive analysis, behavioral calibration, product/action
support, policy optimization, policy results, and robustness/uncertainty. All
policy-facing artifacts use `empirical_bayes_price_consistent`.

Use:

```bash
python scripts/reproduce_paper.py --verify-only
python scripts/reproduce_paper.py
```

The first command verifies existing outputs. The second executes fresh notebook
kernels, is expensive, and finishes by verifying Tables 1--5 and Figures 1--2.
The main policy uses a 12-week decision horizon plus a fixed 36-week
no-promotion evaluation tail, fresh-start initialization, and common-origin
ex-ante PPML profiles. Notebook 07's schedule diagnostics are validation
material only.

Run `python -m pytest`, `python scripts/check_repository.py`, and
`python scripts/verify_paper_outputs.py` after a reproduction.

## Execution notes

The Notebook 02 CSV round-trip check verifies values while allowing Pandas to
change a categorical dtype representation on reload. Notebook 06 writes the
candidate-pruning exactness audit consumed by Notebook 07, so no manual audit
step is required between the two notebooks.

On Windows, an isolated launcher may set `IPYTHONDIR` and `JUPYTER_PATH` to
repository-local temporary locations to avoid user-profile permission issues.
Those settings are execution-environment details, not scientific inputs.
