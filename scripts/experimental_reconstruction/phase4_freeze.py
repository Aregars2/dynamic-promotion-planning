"""Build a versioned Phase 4 freeze manifest from externally supplied evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_promotion_planning.experimental_reconstruction import build_freeze_manifest, read_historical_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    constraints = read_historical_evidence(args.evidence)
    manifest = build_freeze_manifest(
        output_dir=args.output_dir,
        files=args.files,
        protocol_version=args.protocol_version,
        evidence=constraints.evidence,
    )
    print(f"Freeze status: {manifest['freeze_status']}")


if __name__ == "__main__":
    main()
