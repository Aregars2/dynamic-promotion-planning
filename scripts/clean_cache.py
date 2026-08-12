from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "cache"

parser = ArgumentParser(description="Remove regenerable policy caches.")
parser.add_argument("--apply", action="store_true", help="Actually delete cache contents.")
args = parser.parse_args()

paths = [path for path in CACHE.rglob("*") if path.is_file() and path.name != ".gitkeep"]
for path in paths:
    print(path.relative_to(ROOT))

if not args.apply:
    print(f"\nDry run only: {len(paths)} cache files would be deleted. Use --apply to proceed.")
else:
    for path in paths:
        path.unlink()
    for directory in sorted((path for path in CACHE.rglob("*") if path.is_dir()), reverse=True):
        if directory != CACHE and not any(directory.iterdir()):
            directory.rmdir()
    print(f"Deleted {len(paths)} cache files.")
