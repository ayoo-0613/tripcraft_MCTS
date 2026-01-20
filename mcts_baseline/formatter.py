from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .env import POIBlock, State
from .io import TripCraftRow
from .kb import UnifiedKB
from .ollama_client import OllamaClient


_ATTRACTIONS_DATA = None


def _load_attractions_data():
    global _ATTRACTIONS_DATA
    if _ATTRACTIONS_DATA is not None:
        return _ATTRACTIONS_DATA
    try:
        import pandas as pd
        from utils.paths import tripcraft_db_root

        _ATTRACTIONS_DATA = pd.read_csv(tripcraft_db_root() / "attraction" / "cleaned_attractions_final.csv")
    except Exception:
        _ATTRACTIONS_DATA = None
    return _ATTRACTIONS_DATA


def _to_minutes(hhmm: str) -> int:
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 0


def _parse_hhmm(hhmm: str) -> Optional[int]:
    try:
        h, m = hhmm.split(":")
        h_i = int(h)
        m_i = int(m)
    except Exception:
        return None
    if 0 <= h_i <= 23 and 0 <= m_i <= 59:
        return h_i * 60 + m_i
    return None


def _minutes_to_hhmm(minutes: int) -> str:
    minutes = max(0, min(int(minutes), 23 * 60 + 59))
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def build_poi_list_str(poi_blocks: List[POIBlock]) -> str:
    """
    Join as:
      "{name}, {kind} from {start} to {end}, nearest transit: {stop}, {dist}m away; ..."
    Ensure semicolon separated and ends with '.' optional.
    """
    blocks = sorted(poi_blocks, key=lambda b: _to_minutes(b.start))
    parts: List[str] = []
    for b in blocks:
        label = f"{b.name}, {b.kind}"
        parts.append(
            f"{label} from {b.start} to {b.end}, nearest transit: {b.nearest_transit}, {float(b.dist_m):.2f}m away"
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


def _get_mu_d_type(attraction: str, city: str) -> Optional[float]:
    df = _load_attractions_data()
    if df is None or df.empty:
        return None
    attraction_norm = str(attraction or "").strip().lower()
    city_norm = str(city or "").strip().lower()
    if not attraction_norm:
        return None
    if city_norm:
        match = df[
            (df["City"].astype(str).str.strip().str.lower() == city_norm)
            & (df["name"].astype(str).str.strip().str.lower() == attraction_norm)
        ]
    else:
        match = df[df["name"].astype(str).str.strip().str.lower() == attraction_norm]
    if not match.empty:
        try:
            return float(match.iloc[0]["visit_duration"])
        except Exception:
            return None
    return None


def _meal_time_window(meal: str) -> Optional[Tuple[str, str]]:
    params = {
        "breakfast": (9.84, 50.71 / 60),
        "lunch": (14.44, 59.19 / 60),
        "dinner": (20.42, 69.27 / 60),
    }
    if meal not in params:
        return None
    mean_time, mean_duration = params[meal]
    start_h = mean_time - (mean_duration / 2.0)
    end_h = mean_time + (mean_duration / 2.0)
    start_min = int(round(start_h * 60))
    end_min = int(round(end_h * 60))
    if end_min <= start_min:
        end_min = start_min + 30
    return _minutes_to_hhmm(start_min), _minutes_to_hhmm(end_min)


def _attraction_duration_hours(mu_d_type: Optional[float], persona: str, num_attractions: int) -> Optional[float]:
    if mu_d_type is None:
        return None
    k = 16.61 / 60
    mu_d_max = 4
    mu_d_min = 0
    if "Adventure Seeker" in (persona or ""):
        return mu_d_type - k * (num_attractions - mu_d_min)
    return mu_d_type + k * (mu_d_max - num_attractions)


def _default_visit_times(draft: Any, persona: str, kb: UnifiedKB, day: int, city: str) -> Dict[str, Tuple[str, str]]:
    times: Dict[str, Tuple[str, str]] = {}
    for meal in ("breakfast", "lunch", "dinner"):
        name = getattr(draft, meal, "-")
        if name and name != "-":
            window = _meal_time_window(meal)
            if window:
                times[name] = window

    attraction_starts = ["11:30", "16:30"]
    attractions = [a for a in (getattr(draft, "attractions", []) or []) if a and a != "-"]
    num_attractions = len(attractions)
    for idx, name in enumerate(attractions):
        if idx >= len(attraction_starts):
            break
        mu_d_type = _get_mu_d_type(name, city)
        if mu_d_type is None:
            mu_d_type = _find_visit_duration(kb, day, name)
        duration_h = _attraction_duration_hours(mu_d_type, persona, num_attractions)
        start_min = _to_minutes(attraction_starts[idx])
        if duration_h is None:
            end_min = start_min + 120
        else:
            end_min = start_min + int(round(duration_h * 60))
        if end_min <= start_min:
            end_min = start_min + 30
        end_min = min(end_min, 23 * 60 + 59)
        times[name] = (_minutes_to_hhmm(start_min), _minutes_to_hhmm(end_min))
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
    city = _stage_city_for_day(kb, day)
    default_times = _default_visit_times(draft, row.persona, kb, day, city)
    for block in draft.poi_blocks:
        if block.kind != "visit":
            continue
        window = default_times.get(block.name)
        if not window:
            continue
        block.start, block.end = window

    if client is None:
        return

    attractions = [a for a in (getattr(draft, "attractions", []) or []) if a and a != "-"]
    attraction_set = set(attractions)
    meal_map = {
        "breakfast": getattr(draft, "breakfast", "-"),
        "lunch": getattr(draft, "lunch", "-"),
        "dinner": getattr(draft, "dinner", "-"),
    }
    items: List[Dict[str, Any]] = []
    for block in draft.poi_blocks:
        item: Dict[str, Any] = {
            "name": block.name,
            "kind": block.kind,
            "default_start": block.start,
            "default_end": block.end,
        }
        if block.kind == "stay":
            item["category"] = "stay"
        else:
            category = "other"
            for meal, meal_name in meal_map.items():
                if meal_name and meal_name != "-" and block.name == meal_name:
                    category = "meal"
                    item["meal"] = meal
                    break
            if category == "other" and block.name in attraction_set:
                category = "attraction"
                mu_d_type = _get_mu_d_type(block.name, city)
                if mu_d_type is None:
                    mu_d_type = _find_visit_duration(kb, day, block.name)
                if mu_d_type is not None:
                    item["visit_duration"] = mu_d_type
            item["category"] = category
        items.append(item)

    state_summary = {
        "day": day,
        "city": city,
        "persona": row.persona or "",
        "num_attractions": len(attractions),
        "attractions": attractions,
    }

    try:
        suggestions = client.get_temporal_schedule(state_summary, items)
    except Exception:
        return
    if not suggestions:
        return

    by_name: Dict[str, Dict[str, str]] = {}
    for item in suggestions:
        name = str(item.get("name") or "")
        start = str(item.get("start") or "")
        end = str(item.get("end") or "")
        if not name or not start or not end:
            continue
        by_name[name] = {"start": start, "end": end}

    for block in draft.poi_blocks:
        if block.kind != "visit":
            continue
        payload = by_name.get(block.name)
        if not payload:
            continue
        start_min = _parse_hhmm(payload["start"])
        end_min = _parse_hhmm(payload["end"])
        if start_min is None or end_min is None or end_min <= start_min:
            continue
        block.start = _minutes_to_hhmm(start_min)
        block.end = _minutes_to_hhmm(end_min)


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
