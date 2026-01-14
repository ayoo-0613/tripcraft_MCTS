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


_GLOBAL_POI_DF = None


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

    def _missing_cuisines(self, state: State) -> set:
        if not self.required_cuisines:
            return set()
        return set(self.required_cuisines) - self._covered_cuisines(state)

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

    def _missing_attraction_types(self, state: State) -> set:
        if not self.required_attraction_types:
            return set()
        return set(self.required_attraction_types) - self._covered_attraction_types(state)

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

    def _slot_conflicts_transport(self, draft: DayDraft, slot: str) -> bool:
        if draft.transport_start_min is None or draft.transport_end_min is None:
            return False
        window = self._slot_window(slot)
        if not window:
            return False
        return _overlaps((draft.transport_start_min, draft.transport_end_min), window, buffer_min=30)

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
                "meta": {"cost": 0.0, "mode": "transfer"},
                "candidates": [f"Transfer, from {frm} to {to}"],
            }
            if not any(self._budget_ok(state, a) for a in actions):
                actions.append(transfer)
            return actions

        if slot == "accommodation":
            cands = topk_accommodations(stage, local_constraint=self.row.local_constraint, k=topk)
            return [{"type": "set_accommodation", "name": c["name"], "meta": c, "candidates": [x["name"] for x in cands]} for c in cands]

        if slot in {"breakfast", "lunch", "dinner"}:
            if self._slot_conflicts_transport(draft, slot):
                return [{"type": f"skip_{slot}", "name": "-", "candidates": ["-"]}]
            missing_cuisines = self._missing_cuisines(state)
            if slot in {"lunch", "dinner"} and not missing_cuisines:
                return [{"type": f"skip_{slot}", "name": "-", "candidates": ["-"]}]
            cands = topk_restaurants(stage, meal=slot, local_constraint=self.row.local_constraint, k=topk)
            cands = [c for c in cands if (c["name"], stage.city) not in used_restaurants]
            if missing_cuisines:
                helpful = [c for c in cands if self._covers_missing_cuisine(stage, c, missing_cuisines)]
                if slot in {"lunch", "dinner"} and not helpful:
                    return [{"type": f"skip_{slot}", "name": "-", "candidates": ["-"]}]
                if helpful:
                    cands = sorted(cands, key=lambda c: 0 if self._covers_missing_cuisine(stage, c, missing_cuisines) else 1)
            if not cands:
                return [{"type": f"skip_{slot}", "name": "-", "candidates": ["-"]}]
            actions: List[Dict[str, Any]] = [
                {"type": f"set_{slot}", "name": c["name"], "meta": c, "candidates": [x["name"] for x in cands]} for c in cands
            ]
            return actions[:topk]

        if slot in {"attraction1", "attraction2"}:
            if self._slot_conflicts_transport(draft, slot):
                return [{"type": f"skip_{slot}", "name": "-", "candidates": ["-"]}]
            missing_types = self._missing_attraction_types(state)
            if slot == "attraction2" and not missing_types:
                return [{"type": f"skip_{slot}", "name": "-", "candidates": ["-"]}]
            cands = topk_attractions(stage, local_constraint=self.row.local_constraint, k=topk)
            cands = [c for c in cands if (c["name"], stage.city) not in used_attractions]
            if missing_types:
                helpful = [c for c in cands if self._covers_missing_attraction(stage, c, missing_types)]
                if slot == "attraction2" and not helpful:
                    return [{"type": f"skip_{slot}", "name": "-", "candidates": ["-"]}]
                if helpful:
                    cands = sorted(cands, key=lambda c: 0 if self._covers_missing_attraction(stage, c, missing_types) else 1)
            if not cands:
                return [{"type": f"skip_{slot}", "name": "-", "candidates": ["-"]}]
            actions = [
                {"type": f"add_{slot}", "name": c["name"], "meta": c, "candidates": [x["name"] for x in cands]} for c in cands
            ]
            return actions[:topk]

        if slot == "end_day":
            return [{"type": "end_day"}]

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

    def _budget_ok(self, state: State, action: Dict[str, Any]) -> bool:
        add = self._cost_of(action)
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
            state.budget_used += self._cost_of(action)
            if state.day > 1:
                prev = state.drafts[state.day - 2].accommodation
                if prev and prev != "-" and not draft.poi_blocks:
                    prev_stage = self._stage_for_day(state.day - 1)
                    stop, dist = _lookup_transit(prev_stage, prev) if prev_stage else ("UNKNOWN", 99999.0)
                    draft.poi_blocks.append(
                        POIBlock(name=prev, kind="stay", start="08:00", end="09:00", nearest_transit=stop, dist_m=dist)
                    )
            state.substep += 1
            return state

        if action["type"] == "set_accommodation":
            draft.accommodation = action["name"]
            state.budget_used += self._cost_of(action)

            # Add stay POIs (baseline fixed schedule)
            start: Optional[str]
            end: Optional[str]
            if state.day == 1:
                start, end = "08:00", "09:00"
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
            times = {"breakfast": ("09:20", "10:30"), "lunch": ("14:30", "15:30"), "dinner": ("19:30", "21:00")}
            start, end = times[meal]
            setattr(draft, meal, action["name"])
            state.budget_used += self._cost_of(action)
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
            start, end = ("11:30", "13:30") if action["type"] == "add_attraction1" else ("16:30", "18:00")
            draft.attractions.append(name)
            stop, dist = _lookup_transit(stage, name) if stage else ("UNKNOWN", 99999.0)
            draft.poi_blocks.append(
                POIBlock(name=name, kind="visit", start=start, end=end, nearest_transit=stop, dist_m=dist)
            )
            state.substep += 1
            return state

        if action["type"] == "end_day":
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
        return float(filled) - dist_pen + budget_reward

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
