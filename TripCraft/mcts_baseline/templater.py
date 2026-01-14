from __future__ import annotations

from typing import Any, Dict

from .io import TripCraftRow


def make_output_record_template(row: TripCraftRow) -> Dict[str, Any]:
    """
    Return outer record with keys idx/JSON/persona/plan
    Plan list length == row.days
    Each day has required keys initialized with '-'/empty placeholders.
    """
    record: Dict[str, Any] = {
        "idx": row.idx,
        "JSON": {
            "org": row.org,
            "dest": row.dest,
            "days": row.days,
            "visiting_city_number": row.visiting_city_number,
            "date": row.date,
            "people_number": row.people_number,
            "local_constraint": row.local_constraint,
            "budget": row.budget,
            "query": row.query,
            "level": row.level,
        },
        "persona": row.persona,
        "plan": [],
    }
    for d in range(1, row.days + 1):
        record["plan"].append(
            {
                "days": d,
                "current_city": "-",
                "transportation": "-",
                "breakfast": "-",
                "attraction": "-",
                "lunch": "-",
                "dinner": "-",
                "accommodation": "-",
                "event": "-",
                "point_of_interest_list": "",
            }
        )
    return record

