from __future__ import annotations

from typing import Any, Dict, List

from .env import POIBlock, State
from .io import TripCraftRow
from .kb import UnifiedKB


def _to_minutes(hhmm: str) -> int:
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 0


def build_poi_list_str(poi_blocks: List[POIBlock]) -> str:
    """
    Join as:
      "{name}, {kind} from {start} to {end}, nearest transit: {stop}, {dist}m away; ..."
    Ensure semicolon separated and ends with '.' optional.
    """
    blocks = sorted(poi_blocks, key=lambda b: _to_minutes(b.start))
    parts: List[str] = []
    for b in blocks:
        parts.append(
            f"{b.name}, {b.kind} from {b.start} to {b.end}, nearest transit: {b.nearest_transit}, {float(b.dist_m):.2f}m away"
        )
    if not parts:
        return ""
    return "; ".join(parts) + "."


def _stage_city_for_day(kb: UnifiedKB, day: int) -> str:
    if not kb.stages:
        return ""
    idx = min((day - 1) // 2, len(kb.stages) - 1)
    return kb.stages[idx].city


def _with_city(name: str, city: str) -> str:
    if not name or name == "-":
        return "-"
    if not city:
        return name
    lower = name.lower().rstrip()
    suffix = f", {city}".lower()
    if lower.endswith(suffix):
        return name
    return f"{name}, {city}"


def _ensure_transport_placeholder(current_city: str, transportation: str) -> str:
    if transportation != "-":
        return transportation
    if "from " in current_city:
        return f"Transfer, {current_city}"
    return transportation


def _select_events(row: TripCraftRow, kb: UnifiedKB) -> Dict[int, str]:
    local = row.local_constraint or {}
    ev = local.get("event")
    if not ev:
        return {}
    if isinstance(ev, str):
        event_types = [ev]
    else:
        event_types = [str(x) for x in ev]

    used = set()
    events_api = _get_events_api()
    day_events: Dict[int, str] = {}
    for et in event_types:
        placed = False
        for day in range(1, row.days + 1):
            if not kb.stages:
                continue
            stage = kb.stages[min((day - 1) // 2, len(kb.stages) - 1)]
            if stage.events is None or stage.events.empty:
                pass
            if "segmentName" not in stage.events.columns or "name" not in stage.events.columns:
                cand = None
            else:
                cand = stage.events[stage.events["segmentName"] == et]
            if cand is not None and not cand.empty:
                name = str(cand.iloc[0]["name"])
                city = stage.city
                key = (name, city)
                if key in used:
                    continue
                day_events[day] = _with_city(name, city)
                used.add(key)
                placed = True
                break
            if events_api:
                try:
                    event_data = events_api.run(stage.city, row.date)
                except Exception:
                    event_data = None
                if event_data is not None and not isinstance(event_data, str) and not event_data.empty:
                    cand2 = event_data[event_data["segmentName"] == et] if "segmentName" in event_data.columns else None
                    if cand2 is not None and not cand2.empty:
                        name = str(cand2.iloc[0]["name"])
                        city = stage.city
                        key = (name, city)
                        if key in used:
                            continue
                        day_events[day] = _with_city(name, city)
                        used.add(key)
                        placed = True
                        break
        if not placed:
            continue
    return day_events


_EVENTS_API = None


def _get_events_api():
    global _EVENTS_API
    if _EVENTS_API is None:
        try:
            from tools.events.apis import Events

            _EVENTS_API = Events()
        except Exception:
            _EVENTS_API = False
    return _EVENTS_API


def fill_template_with_state(
    template: Dict[str, Any], row: TripCraftRow, kb: UnifiedKB, state: State
) -> Dict[str, Any]:
    """
    Fill template['plan'][d-1] fields:
      current_city, transportation, breakfast/lunch/dinner, accommodation, attraction, event, point_of_interest_list
    Return record ready to dump jsonl.
    """
    day_event_map = _select_events(row, kb)

    for d in range(1, row.days + 1):
        day = template["plan"][d - 1]
        draft = state.drafts[d - 1]
        city = _stage_city_for_day(kb, d)
        day["current_city"] = draft.current_city if draft.current_city != "-" else day["current_city"]
        day["transportation"] = _ensure_transport_placeholder(day["current_city"], draft.transportation)
        day["breakfast"] = _with_city(draft.breakfast, city)
        day["lunch"] = _with_city(draft.lunch, city)
        day["dinner"] = _with_city(draft.dinner, city)
        day["accommodation"] = _with_city(draft.accommodation, city)
        day["event"] = day_event_map.get(d, draft.event)
        if draft.attractions:
            day["attraction"] = "; ".join(_with_city(a, city) for a in draft.attractions)
        else:
            day["attraction"] = "-"
        day["point_of_interest_list"] = build_poi_list_str(draft.poi_blocks)

    return template
