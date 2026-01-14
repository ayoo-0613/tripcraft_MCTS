from __future__ import annotations

from typing import Any, Dict, List

REQUIRED_DAY_KEYS = {
    "days",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
    "event",
    "point_of_interest_list",
}


def validate_record(rec: Dict[str, Any]) -> List[str]:
    """
    Check:
      - has keys idx/JSON/persona/plan
      - plan length equals JSON['days']
      - each day has REQUIRED_DAY_KEYS
      - days field is 1..N
    Return list of errors.
    """
    errors: List[str] = []
    for k in ("idx", "JSON", "persona", "plan"):
        if k not in rec:
            errors.append(f"missing top-level key: {k}")

    if "JSON" in rec and isinstance(rec["JSON"], dict):
        days = rec["JSON"].get("days")
    else:
        days = None

    plan = rec.get("plan")
    if not isinstance(plan, list):
        errors.append("plan must be a list")
        return errors

    if isinstance(days, int) and len(plan) != days:
        errors.append(f"plan length {len(plan)} != JSON.days {days}")

    for i, day in enumerate(plan, start=1):
        if not isinstance(day, dict):
            errors.append(f"plan[{i}] must be dict")
            continue
        missing = REQUIRED_DAY_KEYS.difference(day.keys())
        if missing:
            errors.append(f"plan[{i}] missing day keys: {sorted(missing)}")
        if day.get("days") != i:
            errors.append(f"plan[{i}].days must be {i}")

    return errors

