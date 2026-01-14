from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class StageKB:
    city: str
    attractions: pd.DataFrame
    restaurants: pd.DataFrame
    accommodations: pd.DataFrame
    poi2transit: pd.DataFrame
    events: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class TransportOption:
    mode: str  # "flight" | "self-driving" | "taxi"
    frm: str
    to: str
    date: Optional[str]
    raw: str  # exact string to output in 'transportation'
    cost: Optional[float] = None
    duration_min: Optional[float] = None


@dataclass
class UnifiedKB:
    stages: List[StageKB]  # ordered cities
    transports: List[TransportOption]
