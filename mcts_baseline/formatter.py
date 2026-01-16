from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .env import POIBlock, State
from .io import TripCraftRow
from .kb import UnifiedKB
from .ollama_client import OllamaClient


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


def _parse_hhmm(value: str) -> Optional[Tuple[int, int]]:
    try:
        h, m = value.split(":")
        hour = int(h)
        minute = int(m)
    except Exception:
        return None
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour, minute
    return None


def _needs_time_fill(start: str, end: str) -> bool:
    if not isinstance(start, str) or not isinstance(end, str):
        return True
    if _parse_hhmm(start) is None or _parse_hhmm(end) is None:
        return True
    return _to_minutes(end) <= _to_minutes(start)


def _default_visit_times(draft: Any) -> Dict[str, Tuple[str, str]]:
    times: Dict[str, Tuple[str, str]] = {}
    meal_times = {
        "breakfast": ("09:20", "10:30"),
        "lunch": ("14:30", "15:30"),
        "dinner": ("19:30", "21:00"),
    }
    for meal, window in meal_times.items():
        name = getattr(draft, meal, "-")
        if name and name != "-":
            times[name] = window
    attraction_windows = [("11:30", "13:30"), ("16:30", "18:00")]
    for idx, name in enumerate(getattr(draft, "attractions", []) or []):
        if idx >= len(attraction_windows):
            break
        if name and name != "-":
            times[name] = attraction_windows[idx]
    return times


def _find_visit_duration(kb: UnifiedKB, day: int, name: str) -> Optional[float]:
    if not name:
        return None
    if not kb.stages:
        return None
    idx = min((day - 1) // 2, len(kb.stages) - 1)
    stage = kb.stages[idx]
    df = stage.attractions
    if df is None or df.empty or "name" not in df.columns:
        return None
    target = str(name).strip().lower()
    hit = df[df["name"].astype(str).str.strip().str.lower() == target]
    if hit.empty or "visit_duration" not in hit.columns:
        return None
    try:
        return float(hit.iloc[0]["visit_duration"])
    except Exception:
        return None


def _apply_temporal_guidance(
    *,
    day: int,
    row: TripCraftRow,
    kb: UnifiedKB,
    draft: Any,
    client: Optional[OllamaClient],
) -> None:
    if not getattr(draft, "poi_blocks", None):
        return

    name_to_type: Dict[str, str] = {}
    if getattr(draft, "breakfast", "-") != "-":
        name_to_type[draft.breakfast] = "breakfast"
    if getattr(draft, "lunch", "-") != "-":
        name_to_type[draft.lunch] = "lunch"
    if getattr(draft, "dinner", "-") != "-":
        name_to_type[draft.dinner] = "dinner"
    for attr in getattr(draft, "attractions", []) or []:
        if attr and attr != "-":
            name_to_type[attr] = "attraction"

    default_times = _default_visit_times(draft)
    for block in draft.poi_blocks:
        if block.kind != "visit":
            continue
        window = default_times.get(block.name)
        if not window:
            continue
        if _needs_time_fill(block.start, block.end):
            block.start, block.end = window

    items: List[Dict[str, Any]] = []
    for block in draft.poi_blocks:
        if block.kind != "visit":
            continue
        poi_type = name_to_type.get(block.name)
        if not poi_type:
            continue
        default_start, default_end = default_times.get(block.name, (block.start, block.end))
        payload: Dict[str, Any] = {
            "name": block.name,
            "type": poi_type,
            "default_start": default_start,
            "default_end": default_end,
        }
        if poi_type == "attraction":
            visit_duration = _find_visit_duration(kb, day, block.name)
            if visit_duration is not None:
                payload["visit_duration"] = visit_duration
        items.append(payload)

    if not items:
        return
    if client is None:
        return

    state_summary = {
        "day": day,
        "persona": row.persona,
        "num_attractions": len(getattr(draft, "attractions", []) or []),
    }
    updates = client.get_temporal_schedule(state_summary, items)
    if not updates:
        return

    update_map = {item["name"]: item for item in updates if isinstance(item, dict) and "name" in item}
    for block in draft.poi_blocks:
        if block.kind != "visit":
            continue
        update = update_map.get(block.name)
        if not update:
            continue
        start = update.get("start")
        end = update.get("end")
        if not isinstance(start, str) or not isinstance(end, str):
            continue
        if _parse_hhmm(start) is None or _parse_hhmm(end) is None:
            continue
        if _to_minutes(end) <= _to_minutes(start):
            continue
        block.start = start
        block.end = end


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
    if not events_api:
        return {}
    day_events: Dict[int, str] = {}
    cache: Dict[str, Any] = {}
    for et in event_types:
        placed = False
        for day in range(1, row.days + 1):
            if not kb.stages:
                continue
            stage = kb.stages[min((day - 1) // 2, len(kb.stages) - 1)]
            if stage.city not in cache:
                try:
                    cache[stage.city] = events_api.run(stage.city, row.date)
                except Exception:
                    cache[stage.city] = None
            event_data = cache.get(stage.city)
            if event_data is None or isinstance(event_data, str) or event_data.empty:
                continue
            cand = event_data[event_data["segmentName"] == et] if "segmentName" in event_data.columns else None
            if cand is None or cand.empty:
                continue
            name = str(cand.iloc[0]["name"])
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
    template: Dict[str, Any],
    row: TripCraftRow,
    kb: UnifiedKB,
    state: State,
    *,
    temporal_client: Optional[OllamaClient] = None,
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
        _apply_temporal_guidance(day=d, row=row, kb=kb, draft=draft, client=temporal_client)
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
