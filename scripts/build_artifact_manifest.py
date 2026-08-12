from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "checksums.csv"
BASES = [ROOT / "data" / "processed", ROOT / "artifacts"]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


rows: list[dict[str, object]] = []
for base in BASES:
    if not base.is_dir():
        continue
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name in {".gitkeep", OUTPUT.name}:
            continue
        if "cache" in path.relative_to(ROOT).parts:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )

manifest = pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(OUTPUT, index=False)
print(manifest.to_string(index=False))
print(f"\nSaved to: {OUTPUT}")
