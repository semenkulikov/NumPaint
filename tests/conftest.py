from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

# Делаем пакет numpaint (из src/) доступным при запуске pytest без PYTHONPATH.
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

