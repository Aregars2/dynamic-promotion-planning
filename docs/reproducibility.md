# Reproducibility protocol

Install Python 3.11+ with `python -m pip install -e ".[dev]" -c constraints.txt`.
The undistributed Dominick's source files must be placed in `data/raw/`.

The canonical scientific order is Notebooks 01--08: sample construction,
demand estimation, descriptive analysis, behavioral calibration, product/action
support, policy optimization, policy results, and robustness/uncertainty. All
policy-facing artifacts use `empirical_bayes_price_consistent`; unversioned and
pre-price-consistent policy outputs are historical provenance only.

Use:

```bash
python scripts/reproduce_paper.py --verify-only
python scripts/reproduce_paper.py
```

The first command verifies existing outputs. The second executes fresh notebook
kernels, is expensive, and finishes by verifying Tables 1--5 and Figures 1--2.
The main policy uses a 12-week decision horizon plus a fixed 36-week
no-promotion evaluation tail, fresh-start initialization, and common-origin
ex-ante PPML profiles. Notebook 07's VDO output is validation material only.

Run `python -m pytest`, `python scripts/check_repository.py`, and
`python scripts/verify_paper_outputs.py` after a reproduction.
