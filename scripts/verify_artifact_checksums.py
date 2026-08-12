from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "checksums.csv"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not MANIFEST.is_file():
    raise FileNotFoundError(f"Checksum manifest not found: {MANIFEST}")

errors: list[str] = []
manifest = pd.read_csv(MANIFEST)
for row in manifest.itertuples(index=False):
    path = ROOT / row.relative_path
    if not path.is_file():
        errors.append(f"Missing: {row.relative_path}")
        continue
    if path.stat().st_size != int(row.size_bytes):
        errors.append(f"Size differs: {row.relative_path}")
        continue
    if file_sha256(path) != row.sha256:
        errors.append(f"SHA-256 differs: {row.relative_path}")

if errors:
    print("Artifact verification failed:")
    for error in errors:
        print(" -", error)
    sys.exit(1)
print(f"Verified {len(manifest)} canonical artifacts.")
