from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def tripcraft_db_root() -> Path:
    """
    Returns the local TripCraft database root directory.

    Resolution order:
      1) $TRIPCRAFT_DB_ROOT (preferred)
      2) $ATP_DATABASE_ROOT (legacy)
      3) repo-relative common locations
    """
    env = os.environ.get("TRIPCRAFT_DB_ROOT") or os.environ.get("ATP_DATABASE_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    root = repo_root()
    candidates = [
        root / "TripCraft" / "TripCraft_database",
        root / "Tripcraft" / "TripCraft_database",
        root / "TripCraft_database",
        root / "ATP_database",
    ]
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "TripCraft database not found. Set $TRIPCRAFT_DB_ROOT to your local TripCraft_database directory. "
            f"Tried: {[str(c) for c in candidates]}"
        )
    return found

