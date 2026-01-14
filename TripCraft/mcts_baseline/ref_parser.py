from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from io import StringIO
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .kb import StageKB, TransportOption, UnifiedKB


def parse_ref_block(ref_block: str) -> List[Dict[str, Any]]:
    """
    ref_block is a string representation of a python list/dict like:
      "[{'Description':..., 'Content':...}, ...]"
    Use ast.literal_eval to parse, normalize into a list of items.
    """
    text = (ref_block or "").strip()
    if text == "":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = json.loads(text)

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def content_to_df(content: str) -> pd.DataFrame:
    """
    Try:
      1) pd.read_csv(StringIO(content), sep=r"\\s+", engine="python")
      2) if failed -> sep=","
    Return empty DF if content blank.
    """
    if content is None:
        return pd.DataFrame()
    text = str(content).strip()
    if text == "":
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO(text), sep=r"\s+", engine="python", on_bad_lines="skip")
    except Exception:
        pass
    try:
        return pd.read_csv(StringIO(text), sep=",", engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


_TRANSPORT_DESC_RE = re.compile(
    r"^(?P<mode>Flight|Self-driving|Taxi)\s+from\s+(?P<frm>.+?)\s+to\s+(?P<to>.+?)(?:\s+on\s+(?P<date>\d{4}-\d{2}-\d{2}))?$",
    re.IGNORECASE,
)


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _parse_bracket_list(text: str) -> Optional[List[Any]]:
    t = text.strip()
    if not (t.startswith("[") and t.endswith("]")):
        return None
    try:
        val = ast.literal_eval(t)
        if isinstance(val, list):
            return val
    except Exception:
        return None
    return None


def _parse_restaurants_table(content: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=["name", "cuisines", "avg_cost", "rating"])
    rows: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        if "[" not in ln or "]" not in ln:
            continue
        lb = ln.find("[")
        rb = ln.find("]", lb)
        name = ln[:lb].strip()
        cuisines_str = ln[lb : rb + 1].strip()
        after = ln[rb + 1 :].strip().split()
        if len(after) < 2:
            continue
        avg_cost = _safe_float(after[0])
        rating = _safe_float(after[1])
        cuisines = _parse_bracket_list(cuisines_str) or []
        rows.append({"name": name, "cuisines": cuisines, "avg_cost": avg_cost, "rating": rating})
    return pd.DataFrame(rows, columns=["name", "cuisines", "avg_cost", "rating"])


def _parse_accommodations_table(content: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(
            columns=["name", "roomType", "pricing", "pricing_value", "max_occupancy", "rating", "house_rules"]
        )
    rows: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        toks = ln.split()
        rating_idx = None
        for i in range(len(toks) - 1, -1, -1):
            if _safe_float(toks[i]) is not None and _safe_int(toks[i - 1]) is not None if i - 1 >= 0 else False:
                rating_idx = i
                break
        if rating_idx is None or rating_idx < 3:
            continue
        rating = _safe_float(toks[rating_idx])
        max_occ = _safe_int(toks[rating_idx - 1])
        pricing = toks[rating_idx - 2]
        room_type = toks[rating_idx - 3]
        name = " ".join(toks[: rating_idx - 3]).strip()
        house_rules = " ".join(toks[rating_idx + 1 :]).strip()

        pricing_value: Optional[float] = None
        if pricing and pricing not in {"N/A", "NA", "None", "-"}:
            pricing_value = _safe_float(pricing.replace("$", ""))

        rows.append(
            {
                "name": name,
                "roomType": room_type,
                "pricing": pricing,
                "pricing_value": pricing_value,
                "max_occupancy": max_occ,
                "rating": rating,
                "house_rules": house_rules,
            }
        )
    return pd.DataFrame(
        rows, columns=["name", "roomType", "pricing", "pricing_value", "max_occupancy", "rating", "house_rules"]
    )


def _parse_attractions_table(content: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(
            columns=["name", "subcategories", "visit_duration", "address", "latitude", "longitude", "website"]
        )
    rows: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        toks = ln.split()
        if len(toks) < 4:
            continue
        website = toks[-1]
        lon = _safe_float(toks[-2])
        lat = _safe_float(toks[-3])
        if lon is None or lat is None:
            continue
        rest = " ".join(toks[:-3]).strip()
        if "[" not in rest or "]" not in rest:
            continue
        lb = rest.find("[")
        rb = rest.find("]", lb)
        name = rest[:lb].strip()
        sub_str = rest[lb : rb + 1].strip()
        after = rest[rb + 1 :].strip()
        if after == "":
            continue
        parts = after.split(maxsplit=1)
        visit_duration = _safe_float(parts[0])
        address = parts[1].strip() if len(parts) > 1 else ""
        subcategories = _parse_bracket_list(sub_str) or []
        rows.append(
            {
                "name": name,
                "subcategories": subcategories,
                "visit_duration": visit_duration,
                "address": address,
                "latitude": lat,
                "longitude": lon,
                "website": website,
            }
        )
    return pd.DataFrame(
        rows, columns=["name", "subcategories", "visit_duration", "address", "latitude", "longitude", "website"]
    )


def _parse_events_table(content: str) -> pd.DataFrame:
    df = content_to_df(content)
    if df.empty:
        return pd.DataFrame(columns=["name", "segmentName", "streetAddress", "dateTitle", "url"])
    cols = [c for c in ["name", "segmentName", "streetAddress", "dateTitle", "url"] if c in df.columns]
    return df[cols].copy()


def _extract_city_from_description(description: str) -> Optional[str]:
    for pat in (
        r"^Attractions in (?P<city>.+)$",
        r"^Restaurants in (?P<city>.+)$",
        r"^Accommodations in (?P<city>.+)$",
        r"^Nearest Public Transit Stop from Point of Interest in (?P<city>.+)$",
        r"^Events within .+ in (?P<city>.+)$",
    ):
        m = re.match(pat, description)
        if m:
            return m.group("city").strip()
    return None


def _parse_transport_option(description: str, content: str) -> Optional[TransportOption]:
    m = _TRANSPORT_DESC_RE.match((description or "").strip())
    if not m:
        return None
    mode = m.group("mode").lower()
    frm = m.group("frm").strip()
    to = m.group("to").strip()
    date = m.group("date")

    if mode == "flight":
        raw = "-"
        cost = None
        duration_min = None
        lines = [ln.strip() for ln in str(content or "").splitlines() if ln.strip()]
        if len(lines) >= 2:
            # Typical header: "Flight Number Price DepTime ArrTime ActualElapsedTime Distance"
            # Typical row:    "F3145312 123 08:02 09:47 105 500"
            toks = lines[1].split()
            if len(toks) >= 4:
                flight_no = toks[0]
                if len(toks) > 1:
                    cost = _safe_float(toks[1])
                dep = toks[2]
                arr = toks[3]
                if len(toks) > 4:
                    duration_min = _safe_float(toks[4])
                if dep == "24:00":
                    dep = "00:00"
                if arr == "24:00":
                    arr = "00:00"
                raw = f"Flight Number: {flight_no}, from {frm} to {to}, Departure Time: {dep}, Arrival Time: {arr}"
        return TransportOption(mode="flight", frm=frm, to=to, date=date, raw=raw, cost=cost, duration_min=duration_min)

    content_line = str(content or "").strip()
    if content_line == "":
        return None
    # Content is usually like: "Self-driving, Duration: ..., Distance: ..., Estimated Cost: $..."
    title = "Self-driving" if mode == "self-driving" else "Taxi"
    rest = content_line
    if rest.lower().startswith(title.lower()):
        rest = rest[len(title) :].lstrip(" ,")
    raw = f"{title}, from {frm} to {to}"
    if rest:
        raw += f", {rest}"
    cost = None
    duration_min = None
    m = re.search(r"Duration:\s*([0-9]+(?:\.[0-9]+)?)\s*mins", content_line)
    if m:
        duration_min = _safe_float(m.group(1))
    m = re.search(r"Estimated Cost:\s*\$?([0-9]+(?:\.[0-9]+)?)", content_line)
    if m:
        cost = _safe_float(m.group(1))
    return TransportOption(mode=mode, frm=frm, to=to, date=date, raw=raw, cost=cost, duration_min=duration_min)


def infer_stage_order(org: str, candidate_cities: Set[str], transports: List[TransportOption]) -> List[str]:
    """
    Build directed edges (frm->to). Find a simple path:
      org -> c1 -> c2 -> ... -> ck
    Prefer edges that exist with any mode; if multiple, tie-break:
      self-driving > flight > taxi (configurable)
    Return ordered cities.
    """
    if not candidate_cities:
        return []
    if len(candidate_cities) == 1:
        return [next(iter(candidate_cities))]

    mode_rank = {"taxi": 1, "flight": 2, "self-driving": 3}
    edge_best: Dict[Tuple[str, str], int] = {}
    for t in transports:
        edge_best[(t.frm, t.to)] = max(edge_best.get((t.frm, t.to), 0), mode_rank.get(t.mode, 0))

    best_perm: Optional[Tuple[str, ...]] = None
    best_score: Tuple[int, int] = (-1, -1)
    cities = sorted(candidate_cities)
    for perm in permutations(cities):
        segments = [(org, perm[0])] + list(zip(perm, perm[1:])) + [(perm[-1], org)]
        exist_count = sum(1 for a, b in segments if (a, b) in edge_best)
        rank_sum = sum(edge_best.get((a, b), 0) for a, b in segments)
        score = (exist_count, rank_sum)
        if score > best_score:
            best_score = score
            best_perm = perm

    return list(best_perm) if best_perm else cities


def build_unified_kb(org: str, ref_blocks: List[str]) -> UnifiedKB:
    """
    - Parse all blocks to items.
    - Route items by Description:
        Attractions/Restaurants/Accommodations/Nearest transit/Events
      and build StageKB per city.
    - Parse transport options from descriptions like:
        'Self-driving from A to B'
        'Taxi from A to B'
        'Flight from A to B on YYYY-MM-DD'
    - Infer ordered stage cities using transport graph.
    """
    city_to_bucket: Dict[str, Dict[str, List[str]]] = {}
    transports: List[TransportOption] = []
    pending_cityless: List[Tuple[str, str]] = []  # (desc, content)

    for blk in ref_blocks:
        for item in parse_ref_block(blk):
            desc = str(item.get("Description") or "").strip()
            content = str(item.get("Content") or "")

            t = _parse_transport_option(desc, content)
            if t is not None:
                transports.append(t)
                continue

            city = _extract_city_from_description(desc)
            if city is None:
                dl = desc.lower()
                if dl.startswith("nearest public transit stop from point of interest") or dl.startswith("nearest public transit stop"):
                    pending_cityless.append(("poi2transit", content))
                elif dl.startswith("events within"):
                    pending_cityless.append(("events", content))
                continue
            bucket = city_to_bucket.setdefault(
                city,
                {"attractions": [], "restaurants": [], "accommodations": [], "poi2transit": [], "events": []},
            )

            if desc.startswith("Attractions in "):
                bucket["attractions"].append(content)
            elif desc.startswith("Restaurants in "):
                bucket["restaurants"].append(content)
            elif desc.startswith("Accommodations in "):
                bucket["accommodations"].append(content)
            elif desc.startswith("Nearest Public Transit Stop from Point of Interest in "):
                bucket["poi2transit"].append(content)
            elif desc.startswith("Events within "):
                bucket["events"].append(content)

    # Attach city-less transit/events for 1-city cases (3-day format).
    if pending_cityless and len(city_to_bucket) == 1:
        only_city = next(iter(city_to_bucket.keys()))
        bucket = city_to_bucket[only_city]
        for kind, content in pending_cityless:
            if kind in bucket:
                bucket[kind].append(content)

    # Parse per-city tables; parse poi2transit after we know POI names.
    candidate_cities = set(city_to_bucket.keys())
    ordered_cities = infer_stage_order(org=org, candidate_cities=candidate_cities, transports=transports)

    stages_unordered: Dict[str, StageKB] = {}
    for city, bucket in city_to_bucket.items():
        attractions = pd.concat([_parse_attractions_table(c) for c in bucket["attractions"]], ignore_index=True)
        restaurants = pd.concat([_parse_restaurants_table(c) for c in bucket["restaurants"]], ignore_index=True)
        accommodations = pd.concat([_parse_accommodations_table(c) for c in bucket["accommodations"]], ignore_index=True)

        poi_names: List[str] = []
        for df in (attractions, restaurants, accommodations):
            if "name" in df.columns and not df.empty:
                poi_names.extend([str(x) for x in df["name"].dropna().tolist()])
        poi_names_sorted = sorted(set(poi_names), key=len, reverse=True)

        poi2_rows: List[Dict[str, Any]] = []
        for content in bucket["poi2transit"]:
            lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
            for ln in lines[1:]:
                toks = ln.split()
                if len(toks) < 4:
                    continue
                dist = _safe_float(toks[-1])
                lon = _safe_float(toks[-2])
                lat = _safe_float(toks[-3])
                if dist is None or lon is None or lat is None:
                    continue
                rest = " ".join(toks[:-3]).strip()
                poi = None
                for cand in poi_names_sorted:
                    if rest.startswith(cand + " "):
                        poi = cand
                        stop = rest[len(cand) :].strip()
                        break
                if poi is None:
                    # Fallback: split on the last two spaces-ish by halves.
                    parts = rest.split(maxsplit=1)
                    poi = parts[0]
                    stop = parts[1] if len(parts) > 1 else ""
                poi2_rows.append(
                    {
                        "PoI": poi,
                        "nearest_stop_name": stop,
                        "nearest_stop_latitude": lat,
                        "nearest_stop_longitude": lon,
                        "nearest_stop_distance": dist,
                    }
                )
        poi2transit = pd.DataFrame(
            poi2_rows,
            columns=[
                "PoI",
                "nearest_stop_name",
                "nearest_stop_latitude",
                "nearest_stop_longitude",
                "nearest_stop_distance",
            ],
        )

        events_df = pd.concat([_parse_events_table(c) for c in bucket["events"]], ignore_index=True) if bucket["events"] else None

        stages_unordered[city] = StageKB(
            city=city,
            attractions=attractions,
            restaurants=restaurants,
            accommodations=accommodations,
            poi2transit=poi2transit,
            events=events_df,
        )

    stages = [stages_unordered[c] for c in ordered_cities if c in stages_unordered]
    return UnifiedKB(stages=stages, transports=transports)
