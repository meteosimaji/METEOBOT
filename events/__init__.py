"""Event package setup."""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@dataclass
class EventInfo:
    name: str
    destination: str
    plus: str | None = None
    pro: str | None = None
    category: str = "Utility"
    example: str | None = None
