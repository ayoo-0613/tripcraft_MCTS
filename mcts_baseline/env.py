from __future__ import annotations

import ast
import copy
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .io import TripCraftRow
from .kb import StageKB, UnifiedKB
from .retrieval import select_transport, topk_accommodations, topk_attractions, topk_restaurants
from .persona_utils import average_persona_similarity, candidate_poi_name, get_persona_embeddings, get_poi_embedding


@dataclass
class POIBlock:
    name: str
    kind: str  # "stay" or "visit"
    start: str  # "HH:MM"
    end: str
    nearest_transit: str
    dist_m: float


@dataclass
class DayDraft:
    current_city: str = "-"
    transportation: str = "-"
    transport_start_min: Optional[int] = None
    transport_end_min: Optional[int] = None
    breakfast: str = "-"
    lunch: str = "-"
    dinner: str = "-"
    attractions: List[str] = field(default_factory=list)
    accommodation: str = "-"
    event: str = "-"
    poi_blocks: List[POIBlock] = field(default_factory=list)


@dataclass
class State:
    day: int
    stage_idx: int
    current_city: str
    drafts: List[DayDraft]
    time_cursor_min: int
    budget_used: float
    done: bool = False
    substep: int = 0  # per-day slot cursor
    action_trace: List[Dict[str, Any]] = field(default_factory=list)


def _to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_hhmm(minutes: int) -> str:
    minutes = max(0, min(int(minutes), 23 * 60 + 59))
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    s = s.replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = re.sub(r"[^a-z0-9\s'\",:&-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_city_suffix(name: str, city: str) -> str:
    n = str(name or "").strip()
    c = str(city or "").strip()
    if not c:
        return n
    nl = _normalize_text(n)
    cl = _normalize_text(c)
    if nl.endswith(", " + cl):
        return n[: -(len(c) + 2)].strip()
    if nl.endswith(" " + cl):
        return n[: -(len(c) + 1)].strip()
    return n


def _as_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val if x is not None]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x is not None]
            except Exception:
                return [val]
        return [val]
    return []


def _parse_transport_window_minutes(transportation: str) -> Optional[Tuple[int, int]]:
    t = str(transportation or "")
    m = re.search(r"Departure Time:\s*(\d{2}:\d{2}).*Arrival Time:\s*(\d{2}:\d{2})", t)
    if m:
        dep = _to_minutes(m.group(1))
        arr = _to_minutes(m.group(2))
        if arr < dep:
            arr += 24 * 60
        return dep, arr
    return None


def _overlaps(a: Tuple[int, int], b: Tuple[int, int], buffer_min: int = 0) -> bool:
    a0, a1 = a
    b0, b1 = b
    return (a0 - buffer_min) < b1 and (b0 - buffer_min) < a1


PERSONA_REWARD_WEIGHT = 2.0
PERSONA_UCT_WEIGHT = 0.2
PERSONA_POOL_K = 20
PERSONA_THRESHOLD_MEAL = 0.2
PERSONA_THRESHOLD_ATTRACTION = 0.2
PERSONA_THRESHOLD_ACCOMMODATION = 0.2


_GLOBAL_POI_DF = None
_TOOLS = None
_COST_INDEX = None


def _load_global_pois():
    global _GLOBAL_POI_DF
    if _GLOBAL_POI_DF is None:
        try:
            import pandas as pd
            from utils.paths import tripcraft_db_root

            path = tripcraft_db_root() / "public_transit_gtfs" / "all_poi_nearest_stops.csv"
            _GLOBAL_POI_DF = pd.read_csv(path)
        except Exception:
            _GLOBAL_POI_DF = None
    return _GLOBAL_POI_DF


def _get_tools():
    global _TOOLS
    if _TOOLS is None:
        try:
            from tools.accommodations.apis import Accommodations
            from tools.flights.apis import Flights
            from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
            from tools.restaurants.apis import Restaurants

            _TOOLS = {
                "flights": Flights(),
                "accommodations": Accommodations(),
                "restaurants": Restaurants(),
                "distance": GoogleDistanceMatrix(),
            }
        except Exception:
            _TOOLS = False
    return _TOOLS


def _pricing_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("price")
    else:
        text = str(value).strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = ast.literal_eval(text)
                raw = parsed.get("price") if isinstance(parsed, dict) else text
            except Exception:
                raw = text
        else:
            raw = text
    raw = str(raw or "").replace("$", "").strip()
    try:
        return float(raw) if raw else None
    except Exception:
        return None


def _build_cost_index(tools: Dict[str, Any]) -> Dict[str, Any]:
    flights = {}
    if tools.get("flights") is not None:
        df = tools["flights"].data
        for _, r in df.iterrows():
            key = (str(r.get("Flight Number")), str(r.get("OriginCityName")), str(r.get("DestCityName")))
            flights[key] = r.get("Price")

    restaurants_by_city: Dict[str, List[Tuple[str, float]]] = {}
    restaurants_exact: Dict[Tuple[str, str], float] = {}
    if tools.get("restaurants") is not None:
        df = tools["restaurants"].data
        for _, r in df.iterrows():
            city = str(r.get("City"))
            name = str(r.get("name"))
            avg_cost = r.get("avg_cost")
            try:
                cost_val = float(avg_cost)
            except Exception:
                continue
            restaurants_by_city.setdefault(city, []).append((name, cost_val))
            restaurants_exact[(city, name)] = cost_val

    accommodations_by_city: Dict[str, List[Tuple[str, Any, Any]]] = {}
    accommodations_exact: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
    if tools.get("accommodations") is not None:
        df = tools["accommodations"].data
        for _, r in df.iterrows():
            city = str(r.get("City"))
            name = str(r.get("name"))
            pricing = r.get("pricing")
            max_occ = r.get("max_occupancy")
            accommodations_by_city.setdefault(city, []).append((name, pricing, max_occ))
            accommodations_exact[(city, name)] = (_pricing_to_float(pricing), max_occ)

    distance_cost: Dict[Tuple[str, str, str], Optional[float]] = {}
    if tools.get("distance") is not None:
        df = tools["distance"].data
        for _, r in df.iterrows():
            origin = str(r.get("origin"))
            dest = str(r.get("destination"))
            duration = r.get("duration_min")
            distance = r.get("distance_km")
            if duration is None or distance is None:
                continue
            try:
                duration_val = float(duration)
                distance_val = float(distance)
            except Exception:
                continue
            if math.isnan(duration_val) or math.isnan(distance_val):
                continue
            if duration_val >= 1440:
                continue
            distance_cost[(origin, dest, "self-driving")] = int(distance_val * 0.05)
            distance_cost[(origin, dest, "taxi")] = int(distance_val)

    return {
        "flights": flights,
        "restaurants_by_city": restaurants_by_city,
        "restaurants_exact": restaurants_exact,
        "accommodations_by_city": accommodations_by_city,
        "accommodations_exact": accommodations_exact,
        "distance_cost": distance_cost,
    }


def _get_cost_index():
    global _COST_INDEX
    if _COST_INDEX is None:
        tools = _get_tools()
        if not tools:
            _COST_INDEX = False
        else:
            _COST_INDEX = _build_cost_index(tools)
    return _COST_INDEX


def _lookup_transit(stage: StageKB, poi_name: str) -> Tuple[str, float]:
    if stage is None:
        df = None
    else:
        df = stage.poi2transit
    if df is not None and not df.empty and "PoI" in df.columns:
        # 1) exact match
        hit = df[df["PoI"] == poi_name]
        if not hit.empty:
            row0 = hit.sort_values(by=["nearest_stop_distance"], ascending=True, na_position="last").iloc[0]
            stop = str(row0.get("nearest_stop_name") or "UNKNOWN")
            try:
                dist_f = float(row0.get("nearest_stop_distance"))
            except Exception:
                dist_f = 99999.0
            return stop, dist_f

        # 2) normalized match (+ strip trailing city)
        target = _normalize_text(_strip_city_suffix(poi_name, stage.city))
        if target != "":
            norm_col = "__norm_poi"
            if norm_col not in df.columns:
                df[norm_col] = df["PoI"].astype(str).apply(lambda x: _normalize_text(_strip_city_suffix(x, stage.city)))

            hit = df[df[norm_col] == target]
            if hit.empty:
                # 3) substring fallback
                hit = df[df[norm_col].apply(lambda x: target in x or x in target)]
            if not hit.empty:
                row0 = hit.sort_values(by=["nearest_stop_distance"], ascending=True, na_position="last").iloc[0]
                stop = str(row0.get("nearest_stop_name") or "UNKNOWN")
                try:
                    dist_f = float(row0.get("nearest_stop_distance"))
                except Exception:
                    dist_f = 99999.0
                return stop, dist_f

    # 4) fallback to global POI dataset
    global_df = _load_global_pois()
    if global_df is None or global_df is False:
        return "UNKNOWN", 99999.0
    if global_df is None or getattr(global_df, "empty", False):
        return "UNKNOWN", 99999.0

    city = stage.city if stage is not None else ""
    df_global = global_df
    if city and "City" in df_global.columns:
        df_global = df_global[df_global["City"] == city].copy()
    if df_global.empty or "PoI" not in df_global.columns:
        return "UNKNOWN", 99999.0

    # exact match in global
    hit = df_global[df_global["PoI"] == poi_name]
    if hit.empty:
        target = _normalize_text(_strip_city_suffix(poi_name, city))
        if target == "":
            return "UNKNOWN", 99999.0
        norm_col = "__norm_poi"
        if norm_col not in df_global.columns:
            df_global[norm_col] = df_global["PoI"].astype(str).apply(lambda x: _normalize_text(_strip_city_suffix(x, city)))
        hit = df_global[df_global[norm_col] == target]
        if hit.empty:
            hit = df_global[df_global[norm_col].apply(lambda x: target in x or x in target)]
    if hit.empty:
        return "UNKNOWN", 99999.0

    row0 = hit.sort_values(by=["nearest_stop_distance"], ascending=True, na_position="last").iloc[0]
    stop = str(row0.get("nearest_stop_name") or "UNKNOWN")
    try:
        dist_f = float(row0.get("nearest_stop_distance"))
    except Exception:
        dist_f = 99999.0
    return stop, dist_f


class TripCraftEnv:
    def __init__(self, row: TripCraftRow, kb: UnifiedKB, topk: int = 5):
        self.row = row
        self.kb = kb
        self.topk = topk
        self.PERSONA_UCT_WEIGHT = PERSONA_UCT_WEIGHT
        self.num_stages = len(kb.stages)
        local = row.local_constraint or {}
        self.required_cuisines = _as_list(local.get("cuisine"))
        self.required_attraction_types = _as_list(local.get("attraction"))
        self._restaurant_cuisines: Dict[Tuple[str, str], set] = {}
        self._attraction_types: Dict[Tuple[str, str], set] = {}
        for stage in kb.stages:
            if stage.restaurants is not None and not stage.restaurants.empty and "name" in stage.restaurants.columns:
                for _, r in stage.restaurants.iterrows():
                    name = str(r.get("name"))
                    cuisines = set(_as_list(r.get("cuisines")))
                    self._restaurant_cuisines[(stage.city, name)] = cuisines
            if stage.attractions is not None and not stage.attractions.empty and "name" in stage.attractions.columns:
                for _, r in stage.attractions.iterrows():
                    name = str(r.get("name"))
                    subcats = set(_as_list(r.get("subcategories")))
                    self._attraction_types[(stage.city, name)] = subcats
        self.day_cuisine_targets, self.day_attraction_targets = self._plan_coverage_targets()

    def clone_state(self, state: State) -> State:
        return copy.deepcopy(state)

    def initial_state(self) -> State:
        drafts = [DayDraft() for _ in range(self.row.days)]
        return State(
            day=1,
            stage_idx=min((1 - 1) // 2, max(self.num_stages - 1, 0)),
            current_city="-",
            drafts=drafts,
            time_cursor_min=0,
            budget_used=0.0,
            done=False,
            substep=0,
        )

    def is_terminal(self, state: State) -> bool:
        return state.done

    def _stage_for_day(self, day: int) -> Optional[StageKB]:
        if self.num_stages == 0:
            return None
        idx = min((day - 1) // 2, self.num_stages - 1)
        return self.kb.stages[idx]

    def _day_is_travel(self, day: int) -> bool:
        return day % 2 == 1

    def _city_movement_for_day(self, day: int) -> Tuple[str, str, str]:
        """
        Returns (current_city_string, from_city, to_city_for_transport)
        """
        stages = [s.city for s in self.kb.stages]
        if not stages:
            return "-", self.row.org, self.row.dest
        last_day = self.row.days
        stage_idx = min((day - 1) // 2, len(stages) - 1)
        if day == 1:
            return f"from {self.row.org} to {stages[0]}", self.row.org, stages[0]
        if day == last_day:
            return f"from {stages[-1]} to {self.row.org}", stages[-1], self.row.org
        if self._day_is_travel(day):
            frm = stages[stage_idx - 1]
            to = stages[stage_idx]
            return f"from {frm} to {to}", frm, to
        return f"from {stages[stage_idx]} to {stages[stage_idx]}", stages[stage_idx], stages[stage_idx]

    def _slot_name(self, state: State) -> str:
        # Travel day has a transport slot at the start.
        travel = self._day_is_travel(state.day)
        last_day = state.day == self.row.days
        slots: List[str] = []
        if travel:
            slots.append("transport")
        if not last_day:
            slots.append("accommodation")
        slots += ["breakfast", "attraction1", "lunch", "attraction2", "dinner", "end_day"]
        if last_day:
            # On return day, lodging is not applicable.
            slots = [s for s in slots if s != "accommodation"]
        return slots[min(state.substep, len(slots) - 1)]

    def _slot_window(self, slot: str) -> Optional[Tuple[int, int]]:
        times = {
            "breakfast": ("09:20", "10:30"),
            "attraction1": ("11:30", "13:30"),
            "lunch": ("14:30", "15:30"),
            "attraction2": ("16:30", "18:00"),
            "dinner": ("19:30", "21:00"),
        }
        if slot not in times:
            return None
        start, end = times[slot]
        return _to_minutes(start), _to_minutes(end)

    def _rerank_candidates_by_persona(self, cands: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        if not cands:
            return cands
        persona_embeddings = get_persona_embeddings(self.row.persona or "")
        if not persona_embeddings:
            return cands
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, cand in enumerate(cands):
            name = str(cand.get("name") or "")
            if not name or name == "-":
                scored.append((-1.0, idx, cand))
                continue
            poi_name = candidate_poi_name(name, kind)
            if not poi_name:
                scored.append((-1.0, idx, cand))
                continue
            poi_emb = get_poi_embedding(poi_name)
            if poi_emb is None:
                scored.append((-1.0, idx, cand))
                continue
            score = average_persona_similarity(poi_emb, persona_embeddings)
            if kind == "stay" and "Luxury Traveler" in (self.row.persona or ""):
                score += self._luxury_bonus(cand)
            scored.append((score, idx, cand))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [cand for _, _, cand in scored]

    def _luxury_bonus(self, cand: Dict[str, Any]) -> float:
        name = str(cand.get("name") or "").lower()
        bonus = 0.0
        for kw in ("resort", "spa", "luxury", "grand", "palace", "boutique", "villa", "hotel"):
            if kw in name:
                bonus += 0.05
        rating = cand.get("rating")
        if isinstance(rating, (int, float)):
            bonus += min(float(rating) / 5.0, 0.2)
        pricing = cand.get("pricing_value")
        if isinstance(pricing, (int, float)):
            bonus += min(float(pricing) / 500.0, 0.3)
        return bonus

    def _score_candidates_by_persona(
        self, cands: List[Dict[str, Any]], kind: str
    ) -> Optional[List[Tuple[float, int, Dict[str, Any]]]]:
        if not cands:
            return None
        persona_embeddings = get_persona_embeddings(self.row.persona or "")
        if not persona_embeddings:
            return None
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, cand in enumerate(cands):
            name = str(cand.get("name") or "")
            if not name or name == "-":
                scored.append((-1.0, idx, cand))
                continue
            poi_name = candidate_poi_name(name, kind)
            if not poi_name:
                scored.append((-1.0, idx, cand))
                continue
            poi_emb = get_poi_embedding(poi_name)
            if poi_emb is None:
                scored.append((-1.0, idx, cand))
                continue
            score = average_persona_similarity(poi_emb, persona_embeddings)
            if kind == "stay" and "Luxury Traveler" in (self.row.persona or ""):
                score += self._luxury_bonus(cand)
            scored.append((score, idx, cand))
        return scored

    def _filter_candidates_by_persona(
        self, cands: List[Dict[str, Any]], kind: str, threshold: float
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        scored = self._score_candidates_by_persona(cands, kind)
        if scored is None:
            return cands, None
        scored.sort(key=lambda t: (-t[0], t[1]))
        top_score = scored[0][0] if scored else None
        filtered = [cand for score, _, cand in scored if score >= threshold]
        if filtered:
            return filtered, top_score
        return [cand for _, _, cand in scored], top_score

    def persona_action_score(self, action: Dict[str, Any]) -> float:
        action_type = action.get("type") or ""
        if action_type.startswith("skip_") or action_type in {"end_day", "set_transport"}:
            return 0.0
        name = action.get("eval_poi_name") or action.get("name") or ""
        if not name or name == "-":
            return 0.0
        kind = "stay" if action_type == "set_accommodation" else "visit"
        poi_name = candidate_poi_name(str(name), kind)
        if not poi_name:
            return 0.0
        persona_embeddings = get_persona_embeddings(self.row.persona or "")
        if not persona_embeddings:
            return 0.0
        poi_emb = get_poi_embedding(poi_name)
        if poi_emb is None:
            return 0.0
        score = average_persona_similarity(poi_emb, persona_embeddings)
        if kind == "stay" and "Luxury Traveler" in (self.row.persona or ""):
            meta = action.get("meta") or {}
            meta_with_name = dict(meta)
            meta_with_name["name"] = name
            score += self._luxury_bonus(meta_with_name)
        return score

    def _persona_reward(self, state: State) -> float:
        persona_embeddings = get_persona_embeddings(self.row.persona or "")
        if not persona_embeddings:
            return 0.0
        total = 0.0
        count = 0
        for draft in state.drafts:
            for block in draft.poi_blocks:
                name = str(block.name or "")
                if not name or name == "-":
                    continue
                kind = "stay" if block.kind == "stay" else "visit"
                poi_name = candidate_poi_name(name, kind)
                if not poi_name:
                    continue
                poi_emb = get_poi_embedding(poi_name)
                if poi_emb is None:
                    continue
                total += average_persona_similarity(poi_emb, persona_embeddings)
                count += 1
        return (total / float(count)) if count > 0 else 0.0

    def _plan_coverage_targets(self) -> Tuple[Dict[int, set], Dict[int, set]]:
        day_to_stage: Dict[int, int] = {}
        stage_days: Dict[int, List[int]] = {}
        for day in range(1, self.row.days + 1):
            idx = min((day - 1) // 2, max(self.num_stages - 1, 0))
            day_to_stage[day] = idx
            stage_days.setdefault(idx, []).append(day)

        def _preferred_day(days: List[int]) -> Optional[int]:
            for d in days:
                if not self._day_is_travel(d) and d != self.row.days:
                    return d
            return days[0] if days else None

        stage_cuisines: List[set] = []
        stage_attractions: List[set] = []
        for stage in self.kb.stages:
            cuisines = set()
            if stage.restaurants is not None and not stage.restaurants.empty and "cuisines" in stage.restaurants.columns:
                for c in stage.restaurants["cuisines"]:
                    cuisines.update(_as_list(c))
            stage_cuisines.append(cuisines)

            attrs = set()
            if stage.attractions is not None and not stage.attractions.empty and "subcategories" in stage.attractions.columns:
                for c in stage.attractions["subcategories"]:
                    attrs.update(_as_list(c))
            stage_attractions.append(attrs)

        day_cuisine_targets: Dict[int, set] = {}
        for cuisine in self.required_cuisines:
            chosen_day = None
            for stage_idx, available in enumerate(stage_cuisines):
                if cuisine in available:
                    chosen_day = _preferred_day(stage_days.get(stage_idx, []))
                    break
            if chosen_day:
                day_cuisine_targets.setdefault(chosen_day, set()).add(cuisine)

        day_attr_targets: Dict[int, set] = {}
        for attr_type in self.required_attraction_types:
            chosen_day = None
            for stage_idx, available in enumerate(stage_attractions):
                if attr_type in available:
                    chosen_day = _preferred_day(stage_days.get(stage_idx, []))
                    break
            if chosen_day:
                day_attr_targets.setdefault(chosen_day, set()).add(attr_type)

        return day_cuisine_targets, day_attr_targets

    def _covered_cuisines(self, state: State) -> set:
        covered = set()
        for i, d in enumerate(state.drafts):
            stage = self._stage_for_day(i + 1)
            if stage is None:
                continue
            city = stage.city
            for meal in (d.breakfast, d.lunch, d.dinner):
                if meal and meal != "-":
                    covered |= self._restaurant_cuisines.get((city, meal), set())
        return covered

    def _covered_cuisines_for_day(self, state: State, day: int) -> set:
        covered = set()
        draft = state.drafts[day - 1]
        stage = self._stage_for_day(day)
        if stage is None:
            return covered
        city = stage.city
        for meal in (draft.breakfast, draft.lunch, draft.dinner):
            if meal and meal != "-":
                covered |= self._restaurant_cuisines.get((city, meal), set())
        return covered

    def _missing_cuisines(self, state: State) -> set:
        if not self.required_cuisines:
            return set()
        return set(self.required_cuisines) - self._covered_cuisines(state)

    def _day_missing_cuisines(self, state: State, day: int) -> set:
        targets = self.day_cuisine_targets.get(day, set())
        if not targets:
            return set()
        return set(targets) - self._covered_cuisines_for_day(state, day)

    def _covered_attraction_types(self, state: State) -> set:
        covered = set()
        for i, d in enumerate(state.drafts):
            stage = self._stage_for_day(i + 1)
            if stage is None:
                continue
            city = stage.city
            for attr in d.attractions:
                if attr and attr != "-":
                    covered |= self._attraction_types.get((city, attr), set())
        return covered

    def _covered_attraction_types_for_day(self, state: State, day: int) -> set:
        covered = set()
        draft = state.drafts[day - 1]
        stage = self._stage_for_day(day)
        if stage is None:
            return covered
        city = stage.city
        for attr in draft.attractions:
            if attr and attr != "-":
                covered |= self._attraction_types.get((city, attr), set())
        return covered

    def _missing_attraction_types(self, state: State) -> set:
        if not self.required_attraction_types:
            return set()
        return set(self.required_attraction_types) - self._covered_attraction_types(state)

    def _day_missing_attraction_types(self, state: State, day: int) -> set:
        targets = self.day_attraction_targets.get(day, set())
        if not targets:
            return set()
        return set(targets) - self._covered_attraction_types_for_day(state, day)

    def _covers_missing_cuisine(self, stage: StageKB, cand: Dict[str, Any], missing: set) -> bool:
        cuisines = self._restaurant_cuisines.get((stage.city, cand.get("name")), set())
        if not cuisines:
            cuisines = set(_as_list(cand.get("cuisines")))
        return bool(missing & cuisines)

    def _covers_missing_attraction(self, stage: StageKB, cand: Dict[str, Any], missing: set) -> bool:
        subcats = self._attraction_types.get((stage.city, cand.get("name")), set())
        if not subcats:
            subcats = set(_as_list(cand.get("subcategories")))
        return bool(missing & subcats)

    def _slot_conflicts_transport(self, state: State, slot: str) -> bool:
        draft = state.drafts[state.day - 1]
        window = self._slot_window(slot)
        if not window:
            return False
        if draft.transport_start_min is None or draft.transport_end_min is None:
            return False
        buffer_min = 30
        if self._day_is_travel(state.day):
            if state.day == self.row.days:
                # Return day: ensure activities finish before departure.
                latest_end = draft.transport_start_min - buffer_min
                if window[1] > latest_end:
                    return True
            else:
                # Travel-to day: ensure activities start after arrival.
                earliest_start = draft.transport_end_min + buffer_min
                if window[0] < earliest_start:
                    return True
        return _overlaps((draft.transport_start_min, draft.transport_end_min), window, buffer_min=buffer_min)

    def _append_stay_if_missing(
        self, draft: DayDraft, stage: Optional[StageKB], name: str, start: str, end: str
    ) -> None:
        if not name or name == "-":
            return
        for b in draft.poi_blocks:
            if b.name == name and b.kind == "stay" and b.start == start and b.end == end:
                return
        stop, dist = _lookup_transit(stage, name) if stage else ("UNKNOWN", 99999.0)
        draft.poi_blocks.append(POIBlock(name=name, kind="stay", start=start, end=end, nearest_transit=stop, dist_m=dist))

    def _fallback_accommodation(self, stage: Optional[StageKB], remaining_budget: float) -> Optional[Dict[str, Any]]:
        if stage is None or stage.accommodations is None or stage.accommodations.empty:
            return None
        df = stage.accommodations
        people = int(self.row.people_number or 1)
        best_within = None
        best_any = None
        for _, r in df.iterrows():
            name = str(r.get("name"))
            price_val = r.get("pricing_value")
            try:
                price = float(price_val) if price_val is not None and str(price_val) != "nan" else None
            except Exception:
                price = None
            max_occ = r.get("max_occupancy")
            try:
                max_occ_int = int(max_occ) if max_occ and int(max_occ) > 0 else 1
            except Exception:
                max_occ_int = 1
            rooms = math.ceil(people / max_occ_int)
            cost = price * rooms if price is not None else None
            meta = {"name": name, "pricing_value": price, "max_occupancy": max_occ_int}

            if cost is not None and cost <= remaining_budget:
                if best_within is None or cost < best_within[0]:
                    best_within = (cost, meta)
            if cost is not None:
                if best_any is None or cost < best_any[0]:
                    best_any = (cost, meta)
            else:
                if best_any is None:
                    best_any = (float("inf"), meta)
        if best_within:
            return best_within[1]
        if best_any:
            return best_any[1]
        return None

    def _ensure_accommodation_for_day(self, state: State, day: int) -> None:
        if day >= self.row.days:
            return
        draft = state.drafts[day - 1]
        if draft.accommodation and draft.accommodation != "-":
            return
        stage = self._stage_for_day(day)
        remaining = float(self.row.budget or 0.0) - state.budget_used
        fallback = self._fallback_accommodation(stage, remaining)
        if not fallback:
            return
        draft.accommodation = fallback["name"]
        state.budget_used += self._action_cost(state, {"type": "set_accommodation", "name": fallback["name"]})
        if day == 1 or not self._day_is_travel(day):
            self._append_stay_if_missing(draft, stage, draft.accommodation, "08:00", "09:00")
        if day != self.row.days:
            self._append_stay_if_missing(draft, stage, draft.accommodation, "22:15", "08:00")

    def _ensure_travel_sequence(self, state: State) -> None:
        day = state.day
        draft = state.drafts[day - 1]
        if day > 1 and self._day_is_travel(day):
            prev = state.drafts[day - 2].accommodation
            if prev and prev != "-":
                prev_stage = self._stage_for_day(day - 1)
                self._append_stay_if_missing(draft, prev_stage, prev, "08:00", "09:00")
        self._ensure_accommodation_for_day(state, day)
        if draft.accommodation and draft.accommodation != "-":
            stage = self._stage_for_day(day)
            if day == 1 or (not self._day_is_travel(day) and day != self.row.days):
                self._append_stay_if_missing(draft, stage, draft.accommodation, "08:00", "09:00")
            if day != self.row.days:
                self._append_stay_if_missing(draft, stage, draft.accommodation, "22:15", "08:00")

    def legal_actions(self, state: State, topk: int = 5) -> List[Dict[str, Any]]:
        stage = self._stage_for_day(state.day)
        if stage is None:
            return [{"type": "end_day"}] if self._slot_name(state) == "end_day" else []
        slot = self._slot_name(state)
        draft = state.drafts[state.day - 1]
        used_restaurants = set()
        used_attractions = set()
        for i, d in enumerate(state.drafts):
            stage_i = self._stage_for_day(i + 1)
            if stage_i is None:
                continue
            city = stage_i.city
            if d.breakfast != "-":
                used_restaurants.add((d.breakfast, city))
            if d.lunch != "-":
                used_restaurants.add((d.lunch, city))
            if d.dinner != "-":
                used_restaurants.add((d.dinner, city))
            for attr in d.attractions:
                if attr != "-":
                    used_attractions.add((attr, city))

        if slot == "transport":
            current_city, frm, to = self._city_movement_for_day(state.day)
            date = self.row.date[state.day - 1] if state.day - 1 < len(self.row.date) else ""
            # Build up to 5 candidate strings from KB transports.
            options = [t for t in self.kb.transports if t.frm == frm and t.to == to and (t.date is None or t.date == date)]
            if not options:
                options = [t for t in self.kb.transports if t.frm == frm and t.to == to]

            transport_constraint = (self.row.local_constraint or {}).get("transportation")
            if transport_constraint == "no flight":
                options = [t for t in options if t.mode != "flight"]
            elif transport_constraint == "no self-driving":
                options = [t for t in options if t.mode != "self-driving"]

            # Prefer taxi over self-driving when both are available to avoid conflicts.
            if any(t.mode == "taxi" for t in options) and any(t.mode == "self-driving" for t in options):
                options = [t for t in options if t.mode != "self-driving"]

            def rank(mode: str) -> int:
                return {"flight": 3, "taxi": 2, "self-driving": 1}.get(mode, 0)

            options = sorted(options, key=lambda t: rank(t.mode), reverse=True)[:topk]
            if not options:
                return [{"type": "set_transport", "raw": "-", "from": frm, "to": to, "candidates": ["-"]}]
            actions = [
                {
                    "type": "set_transport",
                    "raw": t.raw,
                    "from": frm,
                    "to": to,
                    "eval_poi_name": "-",
                    "meta": {"cost": t.cost, "mode": t.mode, "duration_min": t.duration_min},
                    "candidates": [x.raw for x in options],
                }
                for t in options
            ]
            transfer = {
                "type": "set_transport",
                "raw": f"Transfer, from {frm} to {to}",
                "from": frm,
                "to": to,
                "eval_poi_name": "-",
                "meta": {"cost": 0.0, "mode": "transfer"},
                "candidates": [f"Transfer, from {frm} to {to}"],
            }
            if not any(self._budget_ok(state, a) for a in actions):
                actions.append(transfer)
            return actions

        if slot == "accommodation":
            pool_k = max(topk, PERSONA_POOL_K)
            cands = topk_accommodations(stage, local_constraint=self.row.local_constraint, k=pool_k)
            cands, _ = self._filter_candidates_by_persona(cands, "stay", PERSONA_THRESHOLD_ACCOMMODATION)
            cands = self._rerank_candidates_by_persona(cands, "stay")[:topk]
            return [
                {
                    "type": "set_accommodation",
                    "name": c["name"],
                    "eval_poi_name": c["name"],
                    "meta": c,
                    "candidates": [x["name"] for x in cands],
                }
                for c in cands
            ]

        if slot in {"breakfast", "lunch", "dinner"}:
            if self._slot_conflicts_transport(state, slot):
                return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
            missing_cuisines = self._missing_cuisines(state)
            day_missing = self._day_missing_cuisines(state, state.day)
            pool_k = max(topk, PERSONA_POOL_K)
            cands = topk_restaurants(stage, meal=slot, local_constraint=self.row.local_constraint, k=pool_k)
            cands = [c for c in cands if (c["name"], stage.city) not in used_restaurants]
            target_missing = day_missing or missing_cuisines
            if target_missing:
                helpful = [c for c in cands if self._covers_missing_cuisine(stage, c, target_missing)]
                if slot in {"lunch", "dinner"} and not helpful and not cands:
                    return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
                if helpful:
                    cands = sorted(cands, key=lambda c: 0 if self._covers_missing_cuisine(stage, c, target_missing) else 1)
            if not cands:
                return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
            cands, top_score = self._filter_candidates_by_persona(cands, "visit", PERSONA_THRESHOLD_MEAL)
            cands = self._rerank_candidates_by_persona(cands, "visit")
            actions: List[Dict[str, Any]] = [
                {
                    "type": f"set_{slot}",
                    "name": c["name"],
                    "eval_poi_name": c["name"],
                    "meta": c,
                    "candidates": [x["name"] for x in cands],
                }
                for c in cands
            ]
            actions = actions[:topk]
            if not missing_cuisines and not day_missing and top_score is not None and top_score < PERSONA_THRESHOLD_MEAL:
                actions.append({"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]})
            if slot in {"lunch", "dinner"} and not missing_cuisines and not day_missing:
                actions.append({"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]})
            return actions

        if slot in {"attraction1", "attraction2"}:
            if self._slot_conflicts_transport(state, slot):
                return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
            adventure = "Adventure Seeker" in (self.row.persona or "")
            if slot == "attraction2" and not adventure:
                return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
            missing_types = self._missing_attraction_types(state)
            day_missing = self._day_missing_attraction_types(state, state.day)
            pool_k = max(topk, PERSONA_POOL_K)
            cands = topk_attractions(stage, local_constraint=self.row.local_constraint, k=pool_k)
            cands = [c for c in cands if (c["name"], stage.city) not in used_attractions]
            target_missing = day_missing or missing_types
            if target_missing:
                helpful = [c for c in cands if self._covers_missing_attraction(stage, c, target_missing)]
                if slot == "attraction2" and not helpful and not cands:
                    return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
                if helpful:
                    cands = sorted(cands, key=lambda c: 0 if self._covers_missing_attraction(stage, c, target_missing) else 1)
            if not cands:
                return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
            cands, _ = self._filter_candidates_by_persona(cands, "visit", PERSONA_THRESHOLD_ATTRACTION)
            cands = self._rerank_candidates_by_persona(cands, "visit")
            actions = [
                {
                    "type": f"add_{slot}",
                    "name": c["name"],
                    "eval_poi_name": c["name"],
                    "meta": c,
                    "candidates": [x["name"] for x in cands],
                }
                for c in cands
            ]
            return actions[:topk]

        if slot == "end_day":
            return [{"type": "end_day", "eval_poi_name": "-"}]

        return []

    def _cost_of(self, action: Dict[str, Any]) -> float:
        meta = action.get("meta") or {}
        if action["type"] == "set_accommodation":
            v = meta.get("pricing_value")
            if not isinstance(v, (int, float)) or v is None:
                return 0.0
            max_occ = meta.get("max_occupancy")
            try:
                max_occ_int = int(max_occ) if max_occ and int(max_occ) > 0 else 1
            except Exception:
                max_occ_int = 1
            people = int(self.row.people_number or 1)
            rooms = math.ceil(people / max_occ_int)
            return float(v) * rooms
        if action["type"].startswith("set_") and "avg_cost" in meta:
            v = meta.get("avg_cost")
            if not isinstance(v, (int, float)) or v is None:
                return 0.0
            people = int(self.row.people_number or 1)
            return float(v) * people
        if action["type"] == "set_transport":
            v = meta.get("cost")
            if not isinstance(v, (int, float)) or v is None:
                return 0.0
            mode = meta.get("mode") or ""
            people = int(self.row.people_number or 1)
            if mode == "flight":
                return float(v) * people
            if mode == "taxi":
                return float(v) * math.ceil(people / 4)
            if mode == "self-driving":
                return float(v) * math.ceil(people / 5)
            return float(v)
        return 0.0

    def _action_cost(self, state: State, action: Dict[str, Any]) -> float:
        index = _get_cost_index()
        if index:
            people = int(self.row.people_number or 1)
            if action["type"] == "set_transport":
                raw = str(action.get("raw") or "")
                frm = action.get("from") or ""
                to = action.get("to") or ""
                if "Flight Number:" in raw:
                    flight_no = raw.split("Flight Number: ")[1].split(",")[0]
                    price = index["flights"].get((flight_no, frm, to))
                    if price is not None:
                        try:
                            return float(price) * people
                        except Exception:
                            pass
                if "Self-driving" in raw or "Taxi" in raw:
                    mode = "self-driving" if "Self-driving" in raw else "taxi"
                    cost = index["distance_cost"].get((frm, to, mode))
                    if cost is not None:
                        multiplier = math.ceil(people / 5) if mode == "self-driving" else math.ceil(people / 4)
                        return float(cost) * multiplier
            if action["type"] == "set_accommodation":
                stage = self._stage_for_day(state.day)
                if stage is not None:
                    city = stage.city
                    name = action.get("name") or ""
                    exact = index["accommodations_exact"].get((city, name))
                    price = None
                    max_occ = None
                    if exact:
                        price, max_occ = exact
                    if price is None:
                        for cand_name, pricing, cand_occ in index["accommodations_by_city"].get(city, []):
                            if name and name in cand_name:
                                price = _pricing_to_float(pricing)
                                max_occ = cand_occ
                                break
                    if price is not None:
                        try:
                            max_occ_val = int(max_occ) if max_occ else 1
                        except Exception:
                            max_occ_val = 1
                        rooms = math.ceil(people / max_occ_val)
                        return float(price) * rooms
            if action["type"].startswith("set_") and "name" in action:
                meal = action["type"].replace("set_", "")
                if meal in {"breakfast", "lunch", "dinner"}:
                    stage = self._stage_for_day(state.day)
                    if stage is not None:
                        city = stage.city
                        name = action.get("name") or ""
                        cost = index["restaurants_exact"].get((city, name))
                        if cost is None:
                            for cand_name, avg_cost in index["restaurants_by_city"].get(city, []):
                                if name and name in cand_name:
                                    cost = avg_cost
                                    break
                        if cost is not None:
                            return float(cost) * people
        return self._cost_of(action)

    def _budget_ok(self, state: State, action: Dict[str, Any]) -> bool:
        add = self._action_cost(state, action)
        if add <= 0:
            return True
        return (state.budget_used + add) <= float(self.row.budget or 0.0)

    def apply_action(self, state: State, action: Dict[str, Any], *, record_trace: bool = False) -> State:
        if not self._budget_ok(state, action):
            return state

        day_idx = state.day - 1
        stage = self._stage_for_day(state.day)
        draft = state.drafts[day_idx]
        slot = self._slot_name(state)

        if draft.current_city == "-":
            cc, _, _ = self._city_movement_for_day(state.day)
            draft.current_city = cc

        if record_trace:
            state.action_trace.append(
                {
                    "day": state.day,
                    "slot": slot,
                    "action_type": action.get("type"),
                    "chosen": action.get("name") or action.get("raw") or "-",
                    "candidates": action.get("candidates"),
                }
            )

        if action["type"] == "set_transport":
            cc, frm, to = self._city_movement_for_day(state.day)
            draft.current_city = cc
            draft.transportation = action.get("raw", "-")
            tw = _parse_transport_window_minutes(draft.transportation)
            if tw:
                draft.transport_start_min, draft.transport_end_min = tw
            state.budget_used += self._action_cost(state, action)
            if state.day > 1:
                prev = state.drafts[state.day - 2].accommodation
                if prev and prev != "-" and not draft.poi_blocks:
                    prev_stage = self._stage_for_day(state.day - 1)
                    stop, dist = _lookup_transit(prev_stage, prev) if prev_stage else ("UNKNOWN", 99999.0)
                    start_min = _to_minutes("08:00")
                    end_min = _to_minutes("09:00")
                    if draft.transport_start_min is not None:
                        latest_end = draft.transport_start_min - 30
                        if latest_end < end_min:
                            end_min = max(latest_end, 0)
                            duration = min(60, end_min)
                            start_min = max(end_min - duration, 0)
                    draft.poi_blocks.append(
                        POIBlock(
                            name=prev,
                            kind="stay",
                            start=_minutes_to_hhmm(start_min),
                            end=_minutes_to_hhmm(end_min),
                            nearest_transit=stop,
                            dist_m=dist,
                        )
                    )
            state.substep += 1
            return state

        if action["type"] == "set_accommodation":
            draft.accommodation = action["name"]
            state.budget_used += self._action_cost(state, action)

            # Add stay POIs (baseline fixed schedule)
            start: Optional[str]
            end: Optional[str]
            if state.day == 1:
                start_min = _to_minutes("08:00")
                end_min = _to_minutes("09:00")
                if draft.transport_end_min is not None:
                    earliest_start = draft.transport_end_min + 30
                    if earliest_start > start_min:
                        start_min = earliest_start
                        end_min = start_min + 60
                night_start_min = _to_minutes("22:15")
                if start_min >= night_start_min:
                    start, end = None, None
                else:
                    if end_min > night_start_min:
                        end_min = night_start_min
                    start = _minutes_to_hhmm(start_min)
                    end = _minutes_to_hhmm(end_min)
            else:
                # On travel days (except day1), do not add a morning stay in the destination city.
                if self._day_is_travel(state.day):
                    start, end = None, None
                else:
                    start, end = "08:00", "09:00"
            stop, dist = _lookup_transit(stage, draft.accommodation) if stage else ("UNKNOWN", 99999.0)
            if start and end:
                draft.poi_blocks.append(
                    POIBlock(name=draft.accommodation, kind="stay", start=start, end=end, nearest_transit=stop, dist_m=dist)
                )
            if state.day != self.row.days:
                stop2, dist2 = _lookup_transit(stage, draft.accommodation) if stage else ("UNKNOWN", 99999.0)
                night_start = "22:15"
                draft.poi_blocks.append(
                    POIBlock(
                        name=draft.accommodation,
                        kind="stay",
                        start=night_start,
                        end="08:00",
                        nearest_transit=stop2,
                        dist_m=dist2,
                    )
                )
            state.substep += 1
            return state

        if action["type"] in {"set_breakfast", "set_lunch", "set_dinner"}:
            meal = action["type"].replace("set_", "")
            if action["name"] in {draft.breakfast, draft.lunch, draft.dinner}:
                state.substep += 1
                return state
            # Visit times are assigned later in formatter (LLM or default template).
            start = end = "00:00"
            setattr(draft, meal, action["name"])
            state.budget_used += self._action_cost(state, action)
            stop, dist = _lookup_transit(stage, action["name"]) if stage else ("UNKNOWN", 99999.0)
            draft.poi_blocks.append(
                POIBlock(name=action["name"], kind="visit", start=start, end=end, nearest_transit=stop, dist_m=dist)
            )
            state.substep += 1
            return state

        if action["type"].startswith("skip_"):
            state.substep += 1
            return state

        if action["type"] in {"add_attraction1", "add_attraction2"}:
            name = action["name"]
            if name in draft.attractions:
                state.substep += 1
                return state
            # Visit times are assigned later in formatter (LLM or default template).
            start = end = "00:00"
            draft.attractions.append(name)
            stop, dist = _lookup_transit(stage, name) if stage else ("UNKNOWN", 99999.0)
            draft.poi_blocks.append(
                POIBlock(name=name, kind="visit", start=start, end=end, nearest_transit=stop, dist_m=dist)
            )
            state.substep += 1
            return state

        if action["type"] == "end_day":
            self._ensure_travel_sequence(state)
            if state.day >= self.row.days:
                state.done = True
                return state
            state.day += 1
            state.stage_idx = min((state.day - 1) // 2, max(self.num_stages - 1, 0))
            state.substep = 0
            state.current_city = "-"
            return state

        return state

    def evaluate(self, state: State) -> float:
        # Simple reward: prefer filled fields + penalize long transit distances.
        filled = 0
        dist_pen = 0.0
        for d in state.drafts:
            for key in ("breakfast", "lunch", "dinner", "accommodation"):
                if getattr(d, key) != "-":
                    filled += 1
            filled += len(d.attractions)
            if d.transportation != "-":
                filled += 1
            for b in d.poi_blocks:
                dist_pen += float(b.dist_m) / 10000.0

        budget_left = max(float(self.row.budget or 0.0) - state.budget_used, 0.0)
        budget_reward = 0.1 * (1.0 / (1.0 + budget_left / 1000.0))
        persona_reward = self._persona_reward(state)
        return float(filled) - dist_pen + budget_reward + (PERSONA_REWARD_WEIGHT * persona_reward)

    def greedy_rollout(self, state: State, topk: int = 5, *, record_trace: bool = False) -> State:
        s = self.clone_state(state)
        guard = 0
        while not self.is_terminal(s) and guard < 500:
            guard += 1
            actions = self.legal_actions(s, topk=topk)
            # Choose first legal action that respects budget.
            chosen = None
            for a in actions:
                if self._budget_ok(s, a):
                    chosen = a
                    break
            if chosen is None:
                chosen = {"type": "end_day"}
            s = self.apply_action(s, chosen, record_trace=record_trace)
        return s
