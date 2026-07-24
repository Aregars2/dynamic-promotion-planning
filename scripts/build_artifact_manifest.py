from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for base in [ROOT / "data" / "processed", ROOT / "artifacts"]:
    if not base.is_dir():
        continue
    for path in base.rglob("*"):
        if path.is_file() and path.name != ".gitkeep":
            rows.append(
                {
                    "relative_path": str(path.relative_to(ROOT)),
                    "size_mb": path.stat().st_size / (1024 ** 2),
                    "extension": path.suffix.lower(),
                }
            )
manifest = pd.DataFrame(rows, columns=["relative_path", "size_mb", "extension"])
manifest = manifest.sort_values("size_mb", ascending=False).reset_index(drop=True)
out = ROOT / "artifact_manifest.csv"
manifest.to_csv(out, index=False)
print(manifest.to_string(index=False))
print(f"\nSaved to: {out}")
