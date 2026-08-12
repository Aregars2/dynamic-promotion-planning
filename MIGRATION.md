# Migration to the refactored scientific workflow

The refactored repository is intended to replace the working tree on the
`repository-cleanup` branch while preserving its Git history.

## Recommended procedure

1. Commit and push the current `repository-cleanup` branch.
2. Create a new branch:

   ```powershell
   git switch -c scientific-workflow-refactor
   ```

3. Extract the refactored bundle to a temporary directory.
4. Replace the branch working tree while preserving `.git` and local `data/raw/`.
5. Recreate or reuse the virtual environment and run:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
   .\.venv\Scripts\python.exe scripts\check_repository.py
   .\.venv\Scripts\python.exe -m pytest
   ```

6. Run Notebooks 06, 07, and 08 from fresh kernels.
7. Confirm the dynamic-dominance, capacity-monotonicity, and coarse/fine-overlap assertions.
8. Run Notebooks 01--08 from fresh kernels before final submission.
9. Commit the refactor and reproduced outputs.

## Notebook renaming

- `03_paper_sample_and_descriptives.ipynb` → `03_descriptive_analysis.ipynb`
- `07_results_and_boundary_validation.ipynb` → split into:
  - `07_policy_results.ipynb`
  - `08_boundary_refinement.ipynb`

## Repository and package naming

The canonical names are now:

- repository: `dynamic-promotion-planning`;
- Python distribution: `dynamic-promotion-planning`;
- Python import package: `dynamic_promotion_planning`;
- scientific title: *Dynamic Promotion Planning under Demand Displacement*.

Update a local Git remote after renaming the GitHub repository:

```powershell
git remote set-url origin https://github.com/Aregars2/dynamic-promotion-planning.git
git remote -v
```

Active imports should use:

```python
from dynamic_promotion_planning.policy import PlanningSpec
```

The old `price_of_extrapolation` import path is retained only to load existing
serialized artifacts. New code must not depend on it.
