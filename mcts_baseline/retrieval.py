from __future__ import annotations

import ast
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd

from .kb import StageKB, TransportOption, UnifiedKB


_DISTANCE_RERANK_MULTIPLIER = 3


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _room_type_ok(room_type: str, constraint: Optional[str]) -> bool:
    if not constraint:
        return True
    rt = str(room_type or "").lower()
    if constraint == "not shared room":
        return rt != "shared_room"
    if constraint == "shared room":
        return rt == "shared_room"
    if constraint == "private room":
        return rt == "private_room"
    if constraint == "entire room":
        return rt == "entire_room"
    return True


def _house_rule_ok(house_rules: str, constraint: Optional[str]) -> bool:
    if not constraint:
        return True
    rules = str(house_rules or "")
    if constraint == "smoking":
        return "No smoking" not in rules
    if constraint == "parties":
        return "No parties" not in rules
    if constraint == "children under 10":
        return "No children under 10" not in rules
    if constraint == "visitors":
        return "No visitors" not in rules
    if constraint == "pets":
        return "No pets" not in rules
    return True


def _normalize_list(val: Any) -> List[str]:
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


def _sort_num(val: Any, default: float) -> float:
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default


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


def _nearest_distance(df: pd.DataFrame) -> float:
    row0 = df.sort_values(by=["nearest_stop_distance"], ascending=True, na_position="last").iloc[0]
    try:
        return float(row0.get("nearest_stop_distance"))
    except Exception:
        return 99999.0


def _lookup_transit_distance(stage: StageKB, poi_name: str) -> float:
    df = stage.poi2transit if stage is not None else None
    if df is not None and not df.empty and "PoI" in df.columns:
        hit = df[df["PoI"] == poi_name]
        if not hit.empty:
            return _nearest_distance(hit)

        target = _normalize_text(_strip_city_suffix(poi_name, stage.city))
        if target != "":
            norm_col = "__norm_poi"
            if norm_col not in df.columns:
                df[norm_col] = df["PoI"].astype(str).apply(lambda x: _normalize_text(_strip_city_suffix(x, stage.city)))

            hit = df[df[norm_col] == target]
            if hit.empty:
                hit = df[df[norm_col].apply(lambda x: target in x or x in target)]
            if not hit.empty:
                return _nearest_distance(hit)

    return 99999.0


def topk_accommodations(stage: StageKB, local_constraint: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """Return [{'name':..., 'city':..., 'raw':...}] sorted by rating desc then pricing if available."""
    df = stage.accommodations.copy()
    if df.empty or "name" not in df.columns:
        return []
    room_type_constraint = local_constraint.get("room type")
    house_rule_constraint = local_constraint.get("house rule")
    filtered = df
    if "roomType" in filtered.columns:
        filtered = filtered[filtered["roomType"].apply(lambda x: _room_type_ok(x, room_type_constraint))]
    if "house_rules" in filtered.columns:
        filtered = filtered[filtered["house_rules"].apply(lambda x: _house_rule_ok(x, house_rule_constraint))]
    if not filtered.empty:
        df = filtered
    if "rating" in df.columns:
        df["__rating"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        df["__rating"] = 0.0
    if "pricing_value" in df.columns:
        df["__pricing"] = pd.to_numeric(df["pricing_value"], errors="coerce")
    elif "pricing" in df.columns:
        def _pricing_from_raw(val: Any) -> Optional[float]:
            if val is None:
                return None
            if isinstance(val, dict):
                raw = val.get("price")
            else:
                text = str(val).strip()
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

        df["__pricing"] = df["pricing"].apply(_pricing_from_raw)
    else:
        df["__pricing"] = pd.NA

    df = df.sort_values(by=["__rating", "__pricing"], ascending=[False, True], na_position="last")
    pool_k = min(len(df), max(k * _DISTANCE_RERANK_MULTIPLIER, k))
    pool: List[Dict[str, Any]] = []
    for _, r in df.head(pool_k).iterrows():
        name = str(r.get("name"))
        pool.append(
            {
                "name": name,
                "city": stage.city,
                "raw": name,
                "pricing_value": _as_float(r.get("pricing_value") if "pricing_value" in r else None),
                "rating": _as_float(r.get("rating") if "rating" in r else None),
                "max_occupancy": _as_float(r.get("max_occupancy") if "max_occupancy" in r else None),
                "pricing_raw": r.get("pricing") if "pricing" in r else None,
                "_dist": _lookup_transit_distance(stage, name),
                "_rating": _sort_num(r.get("__rating"), 0.0),
                "_pricing": _sort_num(r.get("__pricing"), float("inf")),
            }
        )
    pool.sort(key=lambda c: (c["_dist"], -c["_rating"], c["_pricing"]))
    out: List[Dict[str, Any]] = []
    for cand in pool[:k]:
        cand.pop("_dist", None)
        cand.pop("_rating", None)
        cand.pop("_pricing", None)
        out.append(cand)
    return out


def topk_restaurants(stage: StageKB, meal: str, local_constraint: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """
    Filter by cuisine if local_constraint['cuisine'] is not None.
    Rank: rating desc, then avg_cost asc.
    Return k.
    """
    df = stage.restaurants.copy()
    if df.empty or "name" not in df.columns:
        return []

    if "rating" in df.columns:
        df["__rating"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        df["__rating"] = 0.0
    if "avg_cost" in df.columns:
        df["__avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    else:
        df["__avg_cost"] = pd.NA

    df = df.sort_values(by=["__rating", "__avg_cost"], ascending=[False, True], na_position="last")
    pool_k = min(len(df), max(k * _DISTANCE_RERANK_MULTIPLIER, k))
    pool: List[Dict[str, Any]] = []
    for _, r in df.head(pool_k).iterrows():
        name = str(r.get("name"))
        pool.append(
            {
                "name": name,
                "city": stage.city,
                "raw": name,
                "avg_cost": _as_float(r.get("avg_cost") if "avg_cost" in r else None),
                "rating": _as_float(r.get("rating") if "rating" in r else None),
                "meal": meal,
                "cuisines": _normalize_list(r.get("cuisines")),
                "_dist": _lookup_transit_distance(stage, name),
                "_rating": _sort_num(r.get("__rating"), 0.0),
                "_avg_cost": _sort_num(r.get("__avg_cost"), float("inf")),
            }
        )
    pool.sort(key=lambda c: (c["_dist"], -c["_rating"], c["_avg_cost"]))
    out: List[Dict[str, Any]] = []
    for cand in pool[:k]:
        cand.pop("_dist", None)
        cand.pop("_rating", None)
        cand.pop("_avg_cost", None)
        out.append(cand)
    return out


def topk_attractions(stage: StageKB, local_constraint: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """Rank by visit_duration (or rating if exists). Return k."""
    df = stage.attractions.copy()
    if df.empty or "name" not in df.columns:
        return []
    attr_constraint = local_constraint.get("attraction")
    if attr_constraint:
        if isinstance(attr_constraint, str):
            required = {attr_constraint}
        else:
            required = {str(x) for x in attr_constraint}
        if "subcategories" in df.columns:
            filtered = df[df["subcategories"].apply(lambda x: isinstance(x, list) and any(s in required for s in x))]
            if not filtered.empty:
                df = filtered
    if "visit_duration" in df.columns:
        df["__score"] = pd.to_numeric(df["visit_duration"], errors="coerce")
    elif "rating" in df.columns:
        df["__score"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        df["__score"] = 0.0
    df = df.sort_values(by=["__score"], ascending=[False], na_position="last")
    pool_k = min(len(df), max(k * _DISTANCE_RERANK_MULTIPLIER, k))
    pool: List[Dict[str, Any]] = []
    for _, r in df.head(pool_k).iterrows():
        name = str(r.get("name"))
        pool.append(
            {
                "name": name,
                "city": stage.city,
                "raw": name,
                "visit_duration": _as_float(r.get("visit_duration") if "visit_duration" in r else None),
                "subcategories": _normalize_list(r.get("subcategories")),
                "_dist": _lookup_transit_distance(stage, name),
                "_score": _sort_num(r.get("__score"), 0.0),
            }
        )
    pool.sort(key=lambda c: (c["_dist"], -c["_score"]))
    out: List[Dict[str, Any]] = []
    for cand in pool[:k]:
        cand.pop("_dist", None)
        cand.pop("_score", None)
        out.append(cand)
    return out


def _transport_rank_for_selection(mode: str) -> int:
    # Retrieval preference differs from stage-order inference.
    return {"flight": 3, "self-driving": 2, "taxi": 1}.get(mode, 0)


def select_transport(org_or_cityA: str, cityB: str, kb: UnifiedKB, date: str) -> str:
    """
    Choose transportation string:
      prefer flight if available (non-empty content or transport raw string includes flight number),
      else self-driving,
      else taxi.
    Return string for output plan['transportation'].
    """
    candidates: List[TransportOption] = [
        t for t in kb.transports if t.frm == org_or_cityA and t.to == cityB and (t.date is None or t.date == date)
    ]
    if not candidates:
        candidates = [t for t in kb.transports if t.frm == org_or_cityA and t.to == cityB]
    if not candidates:
        return "-"

    def has_flight_detail(t: TransportOption) -> bool:
        return t.mode == "flight" and "Flight Number:" in (t.raw or "")

    candidates_sorted = sorted(
        candidates,
        key=lambda t: (
            _transport_rank_for_selection(t.mode),
            1 if has_flight_detail(t) else 0,
        ),
        reverse=True,
    )
    chosen = candidates_sorted[0]
    if chosen.mode == "flight" and not has_flight_detail(chosen):
        # Flight exists but no details; prefer non-flight if available.
        for t in candidates_sorted:
            if t.mode != "flight":
                return t.raw
    return chosen.raw
