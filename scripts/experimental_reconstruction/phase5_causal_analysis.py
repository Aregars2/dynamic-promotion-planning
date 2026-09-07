"""Guarded Phase 5 entry point. It never runs without a passed Phase 4 freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_promotion_planning.experimental_reconstruction import require_passed_freeze


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    args = parser.parse_args()
    require_passed_freeze(args.freeze_manifest)
    raise NotImplementedError(
        "The causal-analysis framework is intentionally not implemented against real outcomes in this outcome-blind stage."
    )


if __name__ == "__main__":
    main()
