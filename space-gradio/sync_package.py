"""Copy the `littlekedi` model package into the Space (the Space can't pip-install it from the repo).

    python space/sync_package.py

Run it whenever littlekedi/ changes, before uploading the Space.
"""

import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src, dst = root / "littlekedi", Path(__file__).resolve().parent / "littlekedi"
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
print(f"copied {sorted(p.name for p in dst.glob('*.py'))} -> {dst}")
