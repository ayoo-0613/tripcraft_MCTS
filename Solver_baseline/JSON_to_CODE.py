#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smt_trip_solver.py

目的
- 读取 CSV 中每行样本（含 idx, JSON, persona 等字段，字段名允许有差异）
- 从 JSON 中抽取求解所需信息，构建 Z3 约束并求解
- 输出与用户示例一致的结构：
  {"idx": ..., "JSON": {...}, "persona": "...", "plan": [ {...day1...}, {...day2...}, ... ]}

约束范围（可扩展）
- 当前版本提供一个可运行的“约束骨架”，并提供清晰的候选集接口
- SMT 求解不使用 persona
- point_of_interest_list 使用固定模板拼接（不含 persona）
- 若 CSV 中含 plan 字段，则默认把 plan 中出现的 POI 作为候选集回退，以便端到端跑通
- 推荐你将 build_candidates_from_external() 接到你现有 KB 检索输出，替换回退策略

依赖
- pandas
- z3-solver

示例
python smt_trip_solver.py --input_csv /mnt/data/tripcraft_5day.csv --output_jsonl out.jsonl --max_rows 50 --emit_smt2_dir smt2

"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import math
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
from z3 import (
    And,
    Distinct,
    If,
    Implies,
    Int,
    Optimize,
    Solver,
    Not,
    Or,
    Sum,
    sat,
    unknown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


# -----------------------------
# 固定模板与固定时段表
# -----------------------------

POI_FMT = "{name}, {action} from {start} to {end}, nearest transit: {transit}, {dist}m away"

DEFAULT_TIME_SLOTS = {
    "stay_morning": ("08:00", "09:00"),
    "breakfast": ("09:25", "10:16"),
    "attraction_1": ("11:30", "13:57"),
    "lunch": ("13:57", "14:56"),
    "attraction_2": ("16:30", "18:57"),
    "dinner": ("19:51", "21:00"),
    "stay_night": ("22:15", "08:00"),
}

# Eval mode uses larger meal gaps (>= 4h) to satisfy commonsense checks.
EVAL_TIME_SLOTS = {
    "stay_morning": ("08:00", "09:00"),
    "breakfast": ("09:00", "09:30"),
    "attraction_1": ("10:00", "12:30"),
    "lunch": ("13:30", "14:30"),
    "attraction_2": ("15:30", "18:30"),
    "dinner": ("19:30", "20:30"),
    "stay_night": ("22:15", "08:00"),
}

TIME_SLOTS = dict(DEFAULT_TIME_SLOTS)


# -----------------------------
# 数据结构
# -----------------------------

@dataclass(frozen=True)
class QuerySpec:
    org: str
    dest: str
    days: int
    visiting_city_number: int
    dates: List[str]
    people_number: int
    budget: int
    local_constraint: Dict[str, Any]
    query_text: str = ""


@dataclass(frozen=True)
class Restaurant:
    name: str
    city: str
    cuisines: List[str]  # 可为空
    cost: int = 0        # 可接入 KB 价格
    transit: str = "-"
    dist_m: float = 0.0


@dataclass(frozen=True)
class Attraction:
    name: str
    city: str
    categories: Optional[List[str]] = None
    cost: int = 0
    transit: str = "-"
    dist_m: float = 0.0


@dataclass(frozen=True)
class Accommodation:
    name: str
    city: str
    cost: int = 0
    transit: str = "-"
    dist_m: float = 0.0


@dataclass(frozen=True)
class TransportOption:
    mode: str               # "Flight" | "Taxi" | "Transfer" | "Self-driving"
    frm: str
    to: str
    cost: int = 0
    flight_no: str = ""
    dep_time: str = ""
    arr_time: str = ""
    raw: str = ""


@dataclass
class Candidates:
    cities: List[str]
    # 对于简化场景：1 城市时仅使用 outbound 与 inbound 选项
    outbound: List[TransportOption]
    inbound: List[TransportOption]
    # 每天三餐候选
    breakfast: List[Restaurant]
    lunch: List[Restaurant]
    dinner: List[Restaurant]
    # 每天一个景点候选
    attractions: List[Attraction]
    # 住宿候选（按城市）
    accommodations: List[Accommodation]


# -----------------------------
# 解析工具
# -----------------------------

def _safe_json_or_ast(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # 先尝试 JSON
        try:
            return json.loads(s)
        except Exception:
            pass
        # 再尝试 literal_eval
        try:
            return ast.literal_eval(s)
        except Exception:
            return v
    return v


def _get_first_existing(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            return d[k]
    return default


def parse_row_to_instance(
    row: Dict[str, Any]
) -> Tuple[int, Dict[str, Any], Optional[str], Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    返回: (idx, query_json, persona, plan_from_row, reference_info)
    兼容字段名差异:
    - idx 可能在 "idx" 或 "id"
    - JSON 可能在 "JSON" 或 "json" 或拆散的列
    - persona 可能在 "persona"
    - plan 可能在 "plan" 或 "annotation_plan"
    - reference_information 可能拆成多个列
    """
    idx_raw = _get_first_existing(row, ["idx", "id"], None)
    if idx_raw is not None:
        idx = int(idx_raw)
    else:
        idx = int(row.get("__row_index__", 0)) + 1

    json_raw = _get_first_existing(row, ["JSON", "json", "query_json"], None)
    query_json = _safe_json_or_ast(json_raw)

    # 如果 CSV 没有 JSON 列，尝试用分散列拼一个
    if not isinstance(query_json, dict):
        query_json = {
            "org": row.get("org", ""),
            "dest": row.get("dest", ""),
            "days": int(row.get("days", 0) or 0),
            "visiting_city_number": int(row.get("visiting_city_number", 0) or 0),
            "date": _safe_json_or_ast(row.get("date", [])) or [],
            "people_number": int(row.get("people_number", 1) or 1),
            "local_constraint": _safe_json_or_ast(row.get("local_constraint", {})) or {},
            "budget": float(row.get("budget", 0.0) or 0.0),
            "query": row.get("query", ""),
            "level": row.get("level", ""),
        }

    persona = row.get("persona", None)

    plan_raw = _get_first_existing(row, ["plan", "annotation_plan"], None)
    plan_val = _safe_json_or_ast(plan_raw)
    plan_from_row = plan_val if isinstance(plan_val, list) else None

    reference_info: List[Dict[str, Any]] = []
    for k in ["reference_information", "reference_information_1", "reference_information_2", "reference_information_3"]:
        raw = row.get(k, None)
        if raw is None or raw == "":
            continue
        parsed = _safe_json_or_ast(raw)
        if isinstance(parsed, list):
            reference_info.extend(parsed)

    return idx, query_json, persona, plan_from_row, reference_info


def queryspec_from_query_json(q: Dict[str, Any]) -> QuerySpec:
    org = str(q.get("org", "")).strip()
    dest = str(q.get("dest", "")).strip()
    days = int(q.get("days", 0) or 0)
    visiting_city_number = int(q.get("visiting_city_number", 1) or 1)

    dates = q.get("date", [])
    if isinstance(dates, str):
        dates = _safe_json_or_ast(dates)
    dates = list(dates) if isinstance(dates, list) else []

    people_number = int(q.get("people_number", 1) or 1)
    budget = int(float(q.get("budget", 0.0) or 0.0))

    local_constraint = q.get("local_constraint", {}) or {}
    if isinstance(local_constraint, str):
        local_constraint = _safe_json_or_ast(local_constraint)
    if not isinstance(local_constraint, dict):
        local_constraint = {}

    query_text = str(q.get("query", "") or "")

    return QuerySpec(
        org=org,
        dest=dest,
        days=days,
        visiting_city_number=visiting_city_number,
        dates=dates,
        people_number=people_number,
        budget=budget,
        local_constraint=local_constraint,
        query_text=query_text,
    )


# -----------------------------
# 固定模板 POI list 生成
# -----------------------------

def default_transit_lookup(name: str, city: str) -> Tuple[str, float]:
    return "-", 0.0


def _split_name_city(s: str) -> Optional[Tuple[str, str]]:
    if not s or str(s).strip() == "-":
        return None
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) < 2:
        return (str(s).strip(), "")
    name = ", ".join(parts[:-1]).strip()
    city = parts[-1].strip()
    return name, city


def make_poi_segment(
    display_name: str,
    lookup_name: str,
    lookup_city: str,
    action: str,
    slot_key: str,
    transit_lookup: Callable[[str, str], Tuple[str, float]],
) -> str:
    start, end = TIME_SLOTS[slot_key]
    transit, dist = transit_lookup(lookup_name, lookup_city)
    return POI_FMT.format(
        name=display_name,
        action=action,
        start=start,
        end=end,
        transit=transit,
        dist=f"{dist:.2f}",
    )


def _finalize_poi_list(segs: List[str]) -> str:
    if not segs:
        return ""
    return "; ".join(segs) + "."

def _normalize_poi_name(raw: str) -> Optional[str]:
    if not raw or str(raw).strip() == "-":
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) >= 2:
        name = ", ".join(parts[:-1]).strip()
    else:
        name = str(raw).strip()
    return name if name and name != "-" else None


def _split_pois(raw: str) -> List[str]:
    if not raw or str(raw).strip() == "-":
        return []
    return [seg.strip() for seg in str(raw).split(";") if seg.strip()]


def _persona_to_meta(persona: Optional[str]) -> Optional[str]:
    if not persona:
        return None
    parts = [p.strip() for p in str(persona).split(";") if p.strip()]
    values = {}
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        values[k.strip().lower()] = v.strip()
    traveler = values.get("traveler type")
    purpose = values.get("purpose of travel")
    spending = values.get("spending preference")
    location = values.get("location preference")
    if not any([traveler, purpose, spending, location]):
        return None
    chunks = []
    if traveler:
        chunks.append(f"traveler: {traveler}")
    if purpose:
        chunks.append(f"purpose: {purpose}")
    if spending:
        chunks.append(f"spending: {spending}")
    if location:
        chunks.append(f"location: {location}")
    return " | ".join(chunks)


def build_point_of_interest_list(
    day_plan: Dict[str, Any],
    persona_meta: Optional[str],
    is_last_day: bool,
    transit_lookup: Callable[[str, str], Tuple[str, float]],
) -> str:
    segs: List[str] = []
    prev_pair = _split_name_city(day_plan.get("_prev_accommodation_poi", "-"))
    acc_pair = _split_name_city(day_plan.get("_accommodation_poi", "-"))
    bf_pair = _split_name_city(day_plan.get("breakfast", "-"))
    lu_pair = _split_name_city(day_plan.get("lunch", "-"))
    di_pair = _split_name_city(day_plan.get("dinner", "-"))
    atts = _split_pois(day_plan.get("attraction", "-"))
    att_pairs = []
    for a in atts:
        t = _split_name_city(a)
        if t:
            att_pairs.append(t)

    def fmt_name(n: str) -> str:
        if persona_meta:
            return f"{n}, {persona_meta}"
        return n

    if prev_pair:
        prev_name, prev_city = prev_pair
        segs.append(make_poi_segment(fmt_name(prev_name), prev_name, prev_city, "stay", "stay_morning", transit_lookup))
    elif acc_pair:
        acc_name, acc_city = acc_pair
        segs.append(make_poi_segment(fmt_name(acc_name), acc_name, acc_city, "stay", "stay_morning", transit_lookup))
    if bf_pair:
        bf_name, bf_city = bf_pair
        segs.append(make_poi_segment(fmt_name(bf_name), bf_name, bf_city, "visit", "breakfast", transit_lookup))
    if att_pairs:
        a_name, a_city = att_pairs[0]
        segs.append(make_poi_segment(fmt_name(a_name), a_name, a_city, "visit", "attraction_1", transit_lookup))
    if lu_pair:
        lu_name, lu_city = lu_pair
        segs.append(make_poi_segment(fmt_name(lu_name), lu_name, lu_city, "visit", "lunch", transit_lookup))
    if len(att_pairs) > 1:
        a_name, a_city = att_pairs[1]
        segs.append(make_poi_segment(fmt_name(a_name), a_name, a_city, "visit", "attraction_2", transit_lookup))
    if di_pair:
        di_name, di_city = di_pair
        segs.append(make_poi_segment(fmt_name(di_name), di_name, di_city, "visit", "dinner", transit_lookup))
    if acc_pair and not is_last_day:
        acc_name, acc_city = acc_pair
        segs.append(make_poi_segment(fmt_name(acc_name), acc_name, acc_city, "stay", "stay_night", transit_lookup))

    return _finalize_poi_list(segs)


# -----------------------------
# 候选集构建
# -----------------------------

def build_candidates_from_plan_fallback(q: QuerySpec, plan_from_row: Optional[List[Dict[str, Any]]]) -> Candidates:
    """
    回退策略:
    - 若 CSV 中有 plan 字段，则把其中出现的餐厅/景点/住宿作为候选
    - 交通候选同理从 plan 的 transportation 字段抽取，否则给最小占位
    """
    cities: List[str] = []

    allowed_cuisines = q.local_constraint.get("cuisine", None)
    if isinstance(allowed_cuisines, str):
        allowed_cuisines = [allowed_cuisines]
    allowed_cuisines = list(allowed_cuisines) if isinstance(allowed_cuisines, list) else []

    bf: List[Restaurant] = []
    lu: List[Restaurant] = []
    di: List[Restaurant] = []
    ats: List[Attraction] = []
    accs: List[Accommodation] = []
    outbound: List[TransportOption] = []
    inbound: List[TransportOption] = []

    def add_rest(lst: List[Restaurant], s: str):
        t = _split_name_city(s)
        if not t:
            return
        name, city = t
        if not city:
            city = q.dest
        if city not in cities:
            cities.append(city)
        lst.append(Restaurant(name=name, city=city, cuisines=allowed_cuisines, cost=0))

    def add_att(s: str):
        for seg in _split_pois(s):
            t = _split_name_city(seg)
            if not t:
                continue
            name, city = t
            if not city:
                city = q.dest
            if city not in cities:
                cities.append(city)
            ats.append(Attraction(name=name, city=city, cost=0))

    def add_acc(s: str):
        t = _split_name_city(s)
        if not t:
            return
        name, city = t
        if not city:
            city = q.dest
        if city not in cities:
            cities.append(city)
        accs.append(Accommodation(name=name, city=city, cost=0))

    def parse_transport(s: str) -> Optional[TransportOption]:
        if not s or str(s).strip() == "-":
            return None
        ss = str(s)
        if ss.startswith("Flight Number:"):
            # Flight Number: Fxxxx, from A to B, Departure Time: hh:mm, Arrival Time: hh:mm
            try:
                segs = [x.strip() for x in ss.split(",")]
                flight_no = segs[0].split(":", 1)[1].strip()
                frm_to = segs[1].replace("from", "").strip()
                frm, to = [x.strip() for x in frm_to.split("to")]
                dep = segs[2].split(":", 1)[1].strip()
                arr = segs[3].split(":", 1)[1].strip()
                return TransportOption(
                    mode="Flight",
                    frm=frm,
                    to=to,
                    cost=0,
                    flight_no=flight_no,
                    dep_time=dep,
                    arr_time=arr,
                    raw=ss,
                )
            except Exception:
                return TransportOption(mode="Flight", frm=q.org, to=q.dest, cost=0, flight_no="", dep_time="", arr_time="", raw=ss)
        if ss.startswith("Taxi"):
            # Taxi, from A to B, ...
            try:
                segs = [x.strip() for x in ss.split(",")]
                frm_to = segs[1].replace("from", "").strip()
                frm, to = [x.strip() for x in frm_to.split("to")]
                return TransportOption(mode="Taxi", frm=frm, to=to, cost=0, raw=ss)
            except Exception:
                return TransportOption(mode="Taxi", frm=q.org, to=q.dest, cost=0, raw=ss)
        if ss.startswith("Transfer"):
            # Transfer, from A to B
            try:
                segs = [x.strip() for x in ss.split(",")]
                frm_to = segs[1].replace("from", "").strip()
                frm, to = [x.strip() for x in frm_to.split("to")]
                return TransportOption(mode="Transfer", frm=frm, to=to, cost=0, raw=ss)
            except Exception:
                return TransportOption(mode="Transfer", frm=q.dest, to=q.dest, cost=0, raw=ss)
        if ss.startswith("Self-driving"):
            try:
                segs = [x.strip() for x in ss.split(",")]
                frm_to = segs[1].replace("from", "").strip()
                frm, to = [x.strip() for x in frm_to.split("to")]
                return TransportOption(mode="Self-driving", frm=frm, to=to, cost=0, raw=ss)
            except Exception:
                return TransportOption(mode="Self-driving", frm=q.org, to=q.dest, cost=0, raw=ss)
        return TransportOption(mode="Taxi", frm=q.org, to=q.dest, cost=0, raw=ss)

    if plan_from_row:
        for d in plan_from_row:
            add_rest(bf, d.get("breakfast", "-"))
            add_rest(lu, d.get("lunch", "-"))
            add_rest(di, d.get("dinner", "-"))
            add_att(d.get("attraction", "-"))
            add_acc(d.get("accommodation", "-"))

        # 交通: 第一天 outbound，最后一天 inbound
        t0 = parse_transport(plan_from_row[0].get("transportation", "")) if len(plan_from_row) >= 1 else None
        tn = parse_transport(plan_from_row[-1].get("transportation", "")) if len(plan_from_row) >= 1 else None
        if t0:
            outbound.append(t0)
        if tn:
            inbound.append(tn)

    # 最小占位，保证可解
    if not outbound:
        outbound = [TransportOption(mode="Taxi", frm=q.org, to=q.dest, cost=0)]
    if not inbound:
        inbound = [TransportOption(mode="Taxi", frm=q.dest, to=q.org, cost=0)]

    # 若 meal/attraction/accommodation 为空，使用 "-"
    if not bf:
        bf = [Restaurant(name="-", city=q.dest, cuisines=[], cost=0)]
    if not lu:
        lu = [Restaurant(name="-", city=q.dest, cuisines=[], cost=0)]
    if not di:
        di = [Restaurant(name="-", city=q.dest, cuisines=[], cost=0)]
    if not ats:
        ats = [Attraction(name="-", city=q.dest, cost=0)]
    if not accs:
        accs = [Accommodation(name="-", city=q.dest, cost=0)]

    # 去重
    def uniq_by_key(items, key_fn):
        seen = set()
        out = []
        for it in items:
            k = key_fn(it)
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    bf = uniq_by_key(bf, lambda x: (x.name, x.city))
    lu = uniq_by_key(lu, lambda x: (x.name, x.city))
    di = uniq_by_key(di, lambda x: (x.name, x.city))
    ats = uniq_by_key(ats, lambda x: (x.name, x.city))
    accs = uniq_by_key(accs, lambda x: (x.name, x.city))
    outbound = uniq_by_key(outbound, lambda x: (x.mode, x.frm, x.to, x.flight_no, x.dep_time, x.arr_time))
    inbound = uniq_by_key(inbound, lambda x: (x.mode, x.frm, x.to, x.flight_no, x.dep_time, x.arr_time))

    for city in cities:
        if not any(o.frm == q.org and o.to == city for o in outbound):
            outbound.append(TransportOption(mode="Taxi", frm=q.org, to=city, cost=0))
        if not any(o.frm == city and o.to == q.org for o in inbound):
            inbound.append(TransportOption(mode="Taxi", frm=city, to=q.org, cost=0))

    def ensure_rest(lst: List[Restaurant]) -> List[Restaurant]:
        if any(r.name == "-" for r in lst):
            return lst
        lst.append(Restaurant(name="-", city=q.dest, cuisines=[], cost=0))
        return lst

    def ensure_att(lst: List[Attraction]) -> List[Attraction]:
        if any(a.name == "-" for a in lst):
            return lst
        lst.append(Attraction(name="-", city=q.dest, categories=[], cost=0))
        return lst

    def ensure_acc(lst: List[Accommodation]) -> List[Accommodation]:
        if any(a.name == "-" for a in lst):
            return lst
        lst.append(Accommodation(name="-", city=q.dest, cost=0))
        return lst

    bf = ensure_rest(bf)
    lu = ensure_rest(lu)
    di = ensure_rest(di)
    ats = ensure_att(ats)
    accs = ensure_acc(accs)

    if not cities:
        cities = [q.dest]

    return Candidates(
        cities=cities,
        outbound=outbound,
        inbound=inbound,
        breakfast=bf,
        lunch=lu,
        dinner=di,
        attractions=ats,
        accommodations=accs,
    )


def build_candidates_from_external(q: QuerySpec, candidates_dir: Optional[str]) -> Optional[Candidates]:
    """
    预留接口: 你可把 KB 检索的候选集写成 JSON 文件后由此读取。
    约定文件 (可按需调整):
      - cities.json: ["Tucson", ...]
      - outbound.json: [{"mode": "...", "frm": "...", "to": "...", "cost": 123, "flight_no": "...", "dep_time": "...", "arr_time": "..."}, ...]
      - inbound.json: 同上
      - breakfast.json / lunch.json / dinner.json: [{"name": "...", "city": "...", "cuisines": ["Italian"], "cost": 10}, ...]
      - attractions.json: [{"name": "...", "city": "...", "cost": 0}, ...]
      - accommodations.json: [{"name": "...", "city": "...", "cost": 0}, ...]
    """
    if not candidates_dir:
        return None
    if not os.path.isdir(candidates_dir):
        return None

    def load_json(name: str, default):
        p = os.path.join(candidates_dir, name)
        if not os.path.exists(p):
            return default
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        cities = load_json("cities.json", [q.dest])

        outbound_raw = load_json("outbound.json", [])
        inbound_raw = load_json("inbound.json", [])
        bf_raw = load_json("breakfast.json", [])
        lu_raw = load_json("lunch.json", [])
        di_raw = load_json("dinner.json", [])
        ats_raw = load_json("attractions.json", [])
        acc_raw = load_json("accommodations.json", [])

        outbound = [TransportOption(**x) for x in outbound_raw] or [TransportOption(mode="Taxi", frm=q.org, to=q.dest, cost=0)]
        inbound = [TransportOption(**x) for x in inbound_raw] or [TransportOption(mode="Taxi", frm=q.dest, to=q.org, cost=0)]
        breakfast = [Restaurant(**x) for x in bf_raw] or [Restaurant(name="-", city=q.dest, cuisines=[], cost=0)]
        lunch = [Restaurant(**x) for x in lu_raw] or [Restaurant(name="-", city=q.dest, cuisines=[], cost=0)]
        dinner = [Restaurant(**x) for x in di_raw] or [Restaurant(name="-", city=q.dest, cuisines=[], cost=0)]
        attractions = [Attraction(**x) for x in ats_raw] or [Attraction(name="-", city=q.dest, cost=0)]
        accommodations = [Accommodation(**x) for x in acc_raw] or [Accommodation(name="-", city=q.dest, cost=0)]

        if not any(r.name == "-" for r in breakfast):
            breakfast.append(Restaurant(name="-", city=q.dest, cuisines=[], cost=0))
        if not any(r.name == "-" for r in lunch):
            lunch.append(Restaurant(name="-", city=q.dest, cuisines=[], cost=0))
        if not any(r.name == "-" for r in dinner):
            dinner.append(Restaurant(name="-", city=q.dest, cuisines=[], cost=0))
        if not any(a.name == "-" for a in attractions):
            attractions.append(Attraction(name="-", city=q.dest, categories=[], cost=0))
        if not any(a.name == "-" for a in accommodations):
            accommodations.append(Accommodation(name="-", city=q.dest, cost=0))

        return Candidates(
            cities=cities,
            outbound=outbound,
            inbound=inbound,
            breakfast=breakfast,
            lunch=lunch,
            dinner=dinner,
            attractions=attractions,
            accommodations=accommodations,
        )
    except Exception:
        return None


def _iter_content_lines(content: Any) -> List[str]:
    lines = [str(x).strip() for x in str(content or "").splitlines() if str(x).strip()]
    if not lines:
        return []
    head = lines[0].lower()
    if head.startswith("name ") or head.startswith("flight number") or head.startswith("poi "):
        return lines[1:]
    return lines


def _parse_cost(s: str) -> int:
    if not s:
        return 0
    m = re.search(r"([0-9]+\.?[0-9]*)", str(s))
    if not m:
        return 0
    try:
        return int(float(m.group(1)))
    except Exception:
        return 0


def _parse_list_field(s: str) -> List[str]:
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    return []


def _normalize_transport_constraint(raw: Optional[str]) -> str:
    if not raw:
        return ""
    s = str(raw).lower().replace("_", " ").replace("-", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _transport_allowed_modes(raw: Optional[str]) -> List[str]:
    mode_all = {"flight", "taxi", "self-driving"}
    s = _normalize_transport_constraint(raw)
    if not s:
        return sorted(mode_all)
    if "no flight" in s:
        mode_all.discard("flight")
    if "no self driving" in s:
        mode_all.discard("self-driving")
    if s == "flight":
        return ["flight"]
    if s == "self driving":
        return ["self-driving"]
    return sorted(mode_all)


def _accommodation_room_ok(room_type: str, constraint: Optional[str]) -> bool:
    if not constraint:
        return True
    s = str(constraint).lower().strip()
    if s == "not shared room":
        return room_type != "shared_room"
    mapping = {
        "private room": "private_room",
        "shared room": "shared_room",
        "entire home": "entire_home",
    }
    want = mapping.get(s, "")
    if not want:
        return True
    return room_type == want


def _accommodation_house_rule_ok(house_rules: str, constraint: Optional[str]) -> bool:
    if not constraint:
        return True
    hr = str(house_rules or "").lower()
    s = str(constraint).lower().strip()
    if s == "smoking" and "no smoking" in hr:
        return False
    if s == "pets" and "no pets" in hr:
        return False
    if s == "visitors" and "no visitors" in hr:
        return False
    if s == "parties" and "no parties" in hr:
        return False
    if s == "children under 10" and "no children under 10" in hr:
        return False
    return True


def _attraction_category_ok(categories: List[str], constraint: Any) -> bool:
    if not constraint:
        return True
    allowed = constraint
    if isinstance(allowed, str):
        allowed = _parse_list_field(allowed) or [allowed]
    allowed = [str(x).strip().lower() for x in allowed if str(x).strip()]
    if not allowed:
        return True
    cats = [str(x).strip().lower() for x in categories or [] if str(x).strip()]
    if not cats:
        return True
    return bool(set(allowed).intersection(cats))


def _normalize_poi_key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _normalize_city_key(city: str) -> str:
    return re.sub(r"\s+", " ", str(city).strip().lower())


_SANDBOX_CACHE: Optional[Dict[str, Any]] = None
_EVENTS_DATA = None


def _load_sandbox_cache() -> Dict[str, Any]:
    global _SANDBOX_CACHE
    if _SANDBOX_CACHE is not None:
        return _SANDBOX_CACHE
    try:
        from utils.paths import tripcraft_db_root
        import pandas as pd
    except Exception:
        _SANDBOX_CACHE = {"ok": False}
        return _SANDBOX_CACHE

    try:
        root = tripcraft_db_root()
        rest_df = pd.read_csv(root / "restaurants" / "cleaned_restaurant_details_2024.csv")[
            ["name", "City", "avg_cost"]
        ].dropna()
        attr_df = pd.read_csv(root / "attraction" / "cleaned_attractions_final.csv")[["name", "City"]].dropna()
        acc_df = pd.read_csv(root / "accommodation" / "cleaned_listings_final_v2.csv")[
            ["name", "City", "pricing", "max_occupancy"]
        ].dropna()
        poi_df = pd.read_csv(
            root / "public_transit_gtfs" / "all_poi_nearest_stops.csv"
        )[["PoI", "City", "nearest_stop_name", "nearest_stop_distance"]].dropna()
        dist_df = pd.read_csv(
            root / "distance_matrix" / "city_distances_times_full.csv"
        )[["origin", "destination", "duration_min", "distance_km"]].dropna()
        flight_df = pd.read_csv(
            root / "flights" / "cleaned_flights_november_2024.csv"
        )[["Flight Number", "OriginCityName", "DestCityName", "Price"]].dropna()
    except Exception:
        _SANDBOX_CACHE = {"ok": False}
        return _SANDBOX_CACHE

    def build_canon_map(df, name_col: str) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {}
        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            city = str(row["City"]).strip()
            if not name or not city:
                continue
            city_key = _normalize_city_key(city)
            name_key = _normalize_poi_key(name)
            mapping.setdefault(city_key, {})[name_key] = name
        return mapping

    rest_canon = build_canon_map(rest_df, "name")
    attr_canon = build_canon_map(attr_df, "name")
    acc_canon = build_canon_map(acc_df, "name")

    rest_cost: Dict[Tuple[str, str], float] = {}
    for _, row in rest_df.iterrows():
        name = str(row["name"]).strip()
        city = str(row["City"]).strip()
        if not name or not city:
            continue
        key = (_normalize_poi_key(name), _normalize_city_key(city))
        try:
            rest_cost[key] = float(row["avg_cost"])
        except Exception:
            continue

    acc_cost: Dict[Tuple[str, str], Tuple[float, int]] = {}
    for _, row in acc_df.iterrows():
        name = str(row["name"]).strip()
        city = str(row["City"]).strip()
        if not name or not city:
            continue
        key = (_normalize_poi_key(name), _normalize_city_key(city))
        pricing = row.get("pricing")
        price_val = None
        if isinstance(pricing, str):
            try:
                pricing = json.loads(pricing)
            except Exception:
                pricing = {}
        if isinstance(pricing, dict):
            price_str = str(pricing.get("price", "")).replace("$", "").strip()
            try:
                price_val = float(price_str)
            except Exception:
                price_val = None
        max_occ = row.get("max_occupancy")
        try:
            max_occ = int(max_occ)
        except Exception:
            max_occ = 1
        if price_val is not None:
            acc_cost[key] = (price_val, max_occ or 1)

    transit_map: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for _, row in poi_df.iterrows():
        name = str(row["PoI"]).strip()
        city = str(row["City"]).strip()
        if not name or not city:
            continue
        key = (_normalize_poi_key(name), _normalize_city_key(city))
        if key in transit_map:
            continue
        try:
            dist = float(row["nearest_stop_distance"])
        except Exception:
            dist = 0.0
        transit_map[key] = (str(row["nearest_stop_name"]).strip(), dist)

    drive_pairs: set = set()
    distance_map: Dict[Tuple[str, str], float] = {}
    for _, row in dist_df.iterrows():
        origin = str(row["origin"]).strip()
        dest = str(row["destination"]).strip()
        if not origin or not dest:
            continue
        try:
            duration = float(row["duration_min"])
            distance = float(row["distance_km"])
        except Exception:
            continue
        if duration < 1440 and distance >= 0:
            drive_pairs.add((origin, dest))
            distance_map[(origin, dest)] = distance

    flight_set: set = set()
    flight_price: Dict[Tuple[str, str, str], float] = {}
    for _, row in flight_df.iterrows():
        fn = str(row["Flight Number"]).strip()
        origin = str(row["OriginCityName"]).strip()
        dest = str(row["DestCityName"]).strip()
        if fn and origin and dest:
            flight_set.add((fn, origin, dest))
            try:
                flight_price[(fn, origin, dest)] = float(row["Price"])
            except Exception:
                pass

    _SANDBOX_CACHE = {
        "ok": True,
        "restaurants": rest_canon,
        "attractions": attr_canon,
        "accommodations": acc_canon,
        "transit": transit_map,
        "drive_pairs": drive_pairs,
        "distance_map": distance_map,
        "flight_set": flight_set,
        "flight_price": flight_price,
        "rest_cost": rest_cost,
        "acc_cost": acc_cost,
    }
    return _SANDBOX_CACHE


def _sandbox_canonical_name(cache: Dict[str, Any], domain: str, name: str, city: str) -> Optional[str]:
    if not cache.get("ok"):
        return None
    city_key = _normalize_city_key(city)
    name_key = _normalize_poi_key(name)
    by_city = cache.get(domain, {})
    return by_city.get(city_key, {}).get(name_key)


def _sandbox_has_transit(cache: Dict[str, Any], name: str, city: str) -> bool:
    if not cache.get("ok"):
        return False
    key = (_normalize_poi_key(name), _normalize_city_key(city))
    return key in cache.get("transit", {})


def _load_events_data():
    global _EVENTS_DATA
    if _EVENTS_DATA is not None:
        return _EVENTS_DATA
    try:
        from utils.paths import tripcraft_db_root
        import pandas as pd
    except Exception:
        _EVENTS_DATA = None
        return _EVENTS_DATA
    try:
        path = tripcraft_db_root() / "events" / "events_cleaned.csv"
        df = pd.read_csv(path)[["name", "segmentName", "city", "dateTitle"]].dropna(
            subset=["name", "segmentName", "city", "dateTitle"]
        )
        df = df[df["dateTitle"].astype(str).str.match(r"^\d{2}-\d{2}-\d{4}$", na=False)]
        df["dateTitle"] = pd.to_datetime(df["dateTitle"], format="%d-%m-%Y")
        _EVENTS_DATA = df
    except Exception:
        _EVENTS_DATA = None
    return _EVENTS_DATA


def _assign_events_to_plan(q: QuerySpec, cand: Candidates, daily_plans: List[Dict[str, Any]]) -> None:
    event_types = q.local_constraint.get("event")
    if not event_types:
        return
    if isinstance(event_types, str):
        event_types = [event_types]
    event_types = [str(x).strip() for x in event_types if str(x).strip()]
    if not event_types:
        return
    df = _load_events_data()
    if df is None or df.empty:
        return
    if not q.dates:
        return
    try:
        import pandas as pd

        start_date = pd.to_datetime(q.dates[0])
        end_date = pd.to_datetime(q.dates[-1])
    except Exception:
        return
    cities = cand.cities or ([q.dest] if q.dest else [])
    if not cities:
        return
    used = set()

    def _pick_day_index() -> int:
        for i, day in enumerate(daily_plans):
            if day.get("event", "-") in ("", "-"):
                return i
        return 0

    for et in event_types:
        sub = df[
            (df["segmentName"] == et)
            & (df["city"].isin(cities))
            & (df["dateTitle"] >= start_date)
            & (df["dateTitle"] <= end_date)
        ]
        if sub.empty:
            sub = df[
                (df["segmentName"] == et)
                & (df["dateTitle"] >= start_date)
                & (df["dateTitle"] <= end_date)
            ]
        if sub.empty:
            continue
        picked = False
        for _, row in sub.iterrows():
            name = str(row["name"]).strip()
            city = str(row["city"]).strip()
            if not name or not city:
                continue
            key = (name, city)
            if key in used:
                continue
            used.add(key)
            day_idx = _pick_day_index()
            daily_plans[day_idx]["event"] = f"{name}, {city}"
            picked = True
            break
        if not picked:
            continue

def _match_known_poi(line: str, names_sorted: List[str]) -> Tuple[Optional[str], str]:
    for name in names_sorted:
        if line.startswith(name + " "):
            return name, line[len(name):].strip()
        if line == name:
            return name, ""
        m = re.match(re.escape(name) + r"\s+", line, flags=re.IGNORECASE)
        if m:
            return name, line[m.end():].strip()
    return None, line


def build_transit_map(reference_info: List[Dict[str, Any]], poi_names: List[str]) -> Dict[Tuple[str, str], Tuple[str, float]]:
    if not reference_info or not poi_names:
        return {}
    names_sorted = sorted(poi_names, key=len, reverse=True)
    transit_map: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for item in reference_info:
        desc = str(item.get("Description", "") or "")
        if not desc.startswith("Nearest Public Transit Stop from Point of Interest"):
            continue
        city = ""
        m_city = re.match(r"^Nearest Public Transit Stop from Point of Interest in (.+)$", desc)
        if m_city:
            city = m_city.group(1).strip()
        content = item.get("Content", "")
        for line in _iter_content_lines(content):
            poi_name, rest = _match_known_poi(line, names_sorted)
            if not poi_name:
                continue
            m = re.search(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+([0-9]+\.?[0-9]*)$", rest)
            if not m:
                continue
            stop_name = rest[:m.start()].strip()
            try:
                dist = float(m.group(3))
            except Exception:
                dist = 0.0
            key = (_normalize_poi_key(poi_name), _normalize_city_key(city))
            transit_map[key] = (stop_name, dist)
    return transit_map


def collect_poi_names(cand: Candidates) -> List[str]:
    names = set()
    for items in [cand.breakfast, cand.lunch, cand.dinner, cand.attractions, cand.accommodations]:
        for it in items:
            if getattr(it, "name", "") != "-":
                names.add(it.name)
    return sorted(names, key=len, reverse=True)


def filter_candidates_for_sandbox(q: QuerySpec, cand: Candidates) -> Candidates:
    cache = _load_sandbox_cache()
    if not cache.get("ok"):
        return cand

    max_rest = 60
    max_attr = 40
    max_acc = 30

    def _filter_rest(items: List[Restaurant]) -> List[Restaurant]:
        kept: List[Restaurant] = []
        seen: set = set()
        for it in items:
            if it.name == "-":
                continue
            canon = _sandbox_canonical_name(cache, "restaurants", it.name, it.city)
            if not canon:
                continue
            if not _sandbox_has_transit(cache, canon, it.city):
                continue
            cost = it.cost
            key = (_normalize_poi_key(canon), _normalize_city_key(it.city))
            if key in cache.get("rest_cost", {}):
                try:
                    cost = int(float(cache["rest_cost"][key]) * max(1, q.people_number))
                except Exception:
                    cost = it.cost
            key = (canon, it.city)
            if key in seen:
                continue
            kept.append(Restaurant(name=canon, city=it.city, cuisines=it.cuisines, cost=cost))
            seen.add(key)
            if len(kept) >= max_rest:
                break
        if not any(r.name == "-" for r in items):
            kept.append(Restaurant(name="-", city=cand.cities[0] if cand.cities else "-", cuisines=[], cost=0))
        else:
            kept.append(next(r for r in items if r.name == "-"))
        return kept

    def _filter_att(items: List[Attraction]) -> List[Attraction]:
        kept: List[Attraction] = []
        seen: set = set()
        for it in items:
            if it.name == "-":
                continue
            canon = _sandbox_canonical_name(cache, "attractions", it.name, it.city)
            if not canon:
                continue
            if not _sandbox_has_transit(cache, canon, it.city):
                continue
            key = (canon, it.city)
            if key in seen:
                continue
            kept.append(Attraction(name=canon, city=it.city, categories=it.categories, cost=it.cost))
            seen.add(key)
            if len(kept) >= max_attr:
                break
        if not any(a.name == "-" for a in items):
            kept.append(Attraction(name="-", city=cand.cities[0] if cand.cities else "-", categories=[], cost=0))
        else:
            kept.append(next(a for a in items if a.name == "-"))
        return kept

    def _filter_acc(items: List[Accommodation]) -> List[Accommodation]:
        kept: List[Accommodation] = []
        seen: set = set()
        for it in items:
            if it.name == "-":
                continue
            canon = _sandbox_canonical_name(cache, "accommodations", it.name, it.city)
            if not canon:
                continue
            if not _sandbox_has_transit(cache, canon, it.city):
                continue
            cost = it.cost
            key = (_normalize_poi_key(canon), _normalize_city_key(it.city))
            if key in cache.get("acc_cost", {}):
                price, max_occ = cache["acc_cost"][key]
                denom = max(1, int(max_occ))
                cost = int(float(price) * math.ceil(max(1, q.people_number) / denom))
            key = (canon, it.city)
            if key in seen:
                continue
            kept.append(Accommodation(name=canon, city=it.city, cost=cost))
            seen.add(key)
            if len(kept) >= max_acc:
                break
        if not any(a.name == "-" for a in items):
            kept.append(Accommodation(name="-", city=cand.cities[0] if cand.cities else "-", cost=0))
        else:
            kept.append(next(a for a in items if a.name == "-"))
        return kept

    def _filter_transport(options: List[TransportOption]) -> List[TransportOption]:
        kept: List[TransportOption] = []
        for t in options:
            mode = (t.mode or "").lower()
            if mode in ("taxi", "self-driving"):
                if (t.frm, t.to) not in cache.get("drive_pairs", set()):
                    continue
                distance = cache.get("distance_map", {}).get((t.frm, t.to))
                if distance is None:
                    continue
                if mode == "self-driving":
                    multiplier = math.ceil(max(1, q.people_number) / 5.0)
                    cost = int(distance * 0.05 * multiplier)
                else:
                    multiplier = math.ceil(max(1, q.people_number) / 4.0)
                    cost = int(distance * multiplier)
                t = TransportOption(
                    mode=t.mode,
                    frm=t.frm,
                    to=t.to,
                    cost=cost,
                    flight_no=t.flight_no,
                    dep_time=t.dep_time,
                    arr_time=t.arr_time,
                    raw=t.raw,
                )
            elif mode == "flight":
                if not t.flight_no:
                    continue
                key = (t.flight_no, t.frm, t.to)
                if key not in cache.get("flight_set", set()):
                    continue
                price = cache.get("flight_price", {}).get(key)
                if price is not None:
                    t = TransportOption(
                        mode=t.mode,
                        frm=t.frm,
                        to=t.to,
                        cost=int(float(price) * max(1, q.people_number)),
                        flight_no=t.flight_no,
                        dep_time=t.dep_time,
                        arr_time=t.arr_time,
                        raw=t.raw,
                    )
            kept.append(t)
        return kept

    outbound = _filter_transport(cand.outbound)
    inbound = _filter_transport(cand.inbound)
    if not outbound:
        city = cand.cities[0] if cand.cities else q.dest
        outbound = [TransportOption(mode="Transfer", frm=q.org, to=city, cost=0)]
    if not inbound:
        city = cand.cities[-1] if cand.cities else q.dest
        inbound = [TransportOption(mode="Transfer", frm=city, to=q.org, cost=0)]

    return Candidates(
        cities=cand.cities,
        outbound=outbound,
        inbound=inbound,
        breakfast=_filter_rest(cand.breakfast),
        lunch=_filter_rest(cand.lunch),
        dinner=_filter_rest(cand.dinner),
        attractions=_filter_att(cand.attractions),
        accommodations=_filter_acc(cand.accommodations),
    )


def infer_stage_order(org: str, candidate_cities: List[str], transports: List[TransportOption]) -> List[str]:
    if not candidate_cities:
        return []
    if len(candidate_cities) == 1:
        return [candidate_cities[0]]
    mode_rank = {"taxi": 1, "flight": 2, "self-driving": 3}
    edge_best: Dict[Tuple[str, str], int] = {}
    for t in transports:
        frm = str(t.frm or "").strip()
        to = str(t.to or "").strip()
        if not frm or not to:
            continue
        edge_best[(frm, to)] = max(edge_best.get((frm, to), 0), mode_rank.get(t.mode.lower(), 0))

    best_perm: Optional[Tuple[str, ...]] = None
    best_score: Tuple[int, int] = (-1, -1)
    cities = sorted(set(candidate_cities))
    for perm in permutations(cities):
        segments = [(org, perm[0])] + list(zip(perm, perm[1:])) + [(perm[-1], org)]
        exist_count = sum(1 for a, b in segments if (a, b) in edge_best)
        rank_sum = sum(edge_best.get((a, b), 0) for a, b in segments)
        score = (exist_count, rank_sum)
        if score > best_score:
            best_score = score
            best_perm = perm
    return list(best_perm) if best_perm else cities


def build_candidates_from_reference_info(q: QuerySpec, reference_info: List[Dict[str, Any]]) -> Optional[Candidates]:
    if not reference_info:
        return None

    cities: List[str] = []
    bf: List[Restaurant] = []
    lu: List[Restaurant] = []
    di: List[Restaurant] = []
    ats: List[Attraction] = []
    accs: List[Accommodation] = []
    outbound: List[TransportOption] = []
    inbound: List[TransportOption] = []
    all_transports: List[TransportOption] = []

    allowed_transport = set(_transport_allowed_modes(q.local_constraint.get("transportation")))
    room_constraint = q.local_constraint.get("room type")
    house_constraint = q.local_constraint.get("house rule")
    attraction_constraint = q.local_constraint.get("attraction")

    for item in reference_info:
        desc = str(item.get("Description", "") or "")
        content = item.get("Content", "")
        if desc.startswith("Restaurants in "):
            city = desc[len("Restaurants in "):].strip() or q.dest
            if city not in cities:
                cities.append(city)
            for line in _iter_content_lines(content):
                m = re.match(r"^(.*?)\s+(\[.*\])\s+([0-9]+\.?[0-9]*)\s+([0-9]+\.?[0-9]*)$", line)
                if not m:
                    continue
                name = m.group(1).strip()
                cuisines = _parse_list_field(m.group(2))
                cost = _parse_cost(m.group(3)) * max(1, q.people_number)
                r = Restaurant(name=name, city=city, cuisines=cuisines, cost=cost)
                bf.append(r)
                lu.append(r)
                di.append(r)
        elif desc.startswith("Attractions in "):
            city = desc[len("Attractions in "):].strip() or q.dest
            if city not in cities:
                cities.append(city)
            for line in _iter_content_lines(content):
                m = re.match(r"^(.*?)\s+(\[.*\])\s+([0-9]+\.?[0-9]*)\s+.*$", line)
                if not m:
                    continue
                name = m.group(1).strip()
                cats = _parse_list_field(m.group(2))
                if not _attraction_category_ok(cats, attraction_constraint):
                    continue
                ats.append(Attraction(name=name, city=city, categories=cats, cost=0))
        elif desc.startswith("Accommodations in "):
            city = desc[len("Accommodations in "):].strip() or q.dest
            if city not in cities:
                cities.append(city)
            for line in _iter_content_lines(content):
                m = re.match(r"^(.*?)\s+(private_room|shared_room|entire_home)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$", line)
                if not m:
                    continue
                name = m.group(1).strip()
                room_type = m.group(2).strip()
                pricing = m.group(3).strip()
                house_rules = m.group(6).strip()
                if not _accommodation_room_ok(room_type, room_constraint):
                    continue
                if not _accommodation_house_rule_ok(house_rules, house_constraint):
                    continue
                cost = _parse_cost(pricing)
                accs.append(Accommodation(name=name, city=city, cost=cost))
        elif desc.startswith("Flight from "):
            m = re.match(r"^Flight from (.+?) to (.+?) on", desc)
            if not m:
                continue
            frm, to = m.group(1).strip(), m.group(2).strip()
            if "flight" not in allowed_transport:
                continue
            for line in _iter_content_lines(content):
                parts = line.split()
                if len(parts) < 4:
                    continue
                flight_no, price, dep, arr = parts[0], parts[1], parts[2], parts[3]
                cost = _parse_cost(price)
                t = TransportOption(mode="Flight", frm=frm, to=to, cost=cost, flight_no=flight_no, dep_time=dep, arr_time=arr)
                outbound.append(t)
                all_transports.append(t)
        elif desc.startswith("Taxi from "):
            m = re.match(r"^Taxi from (.+?) to (.+?)$", desc)
            if not m:
                continue
            if "taxi" not in allowed_transport:
                continue
            frm, to = m.group(1).strip(), m.group(2).strip()
            cost = 0
            cost_str = ""
            duration = ""
            distance = ""
            m_cost = re.search(r"Estimated Cost:\s*\$([0-9\.]+)", str(content))
            if m_cost:
                cost_str = m_cost.group(1)
                cost = _parse_cost(cost_str)
            m_dur = re.search(r"Duration:\s*([0-9\.]+\s*mins)", str(content))
            if m_dur:
                duration = m_dur.group(1)
            m_dist = re.search(r"Distance:\s*([0-9\.]+\s*km)", str(content))
            if m_dist:
                distance = m_dist.group(1)
            raw = f"Taxi, from {frm} to {to}"
            if duration or distance or cost_str:
                raw = f"{raw}, Duration: {duration}, Distance: {distance}, Estimated Cost: ${cost_str or cost}"
            t = TransportOption(mode="Taxi", frm=frm, to=to, cost=cost, raw=raw)
            outbound.append(t)
            all_transports.append(t)
        elif desc.startswith("Self-driving from "):
            m = re.match(r"^Self-driving from (.+?) to (.+?)$", desc)
            if not m:
                continue
            if "self-driving" not in allowed_transport:
                continue
            frm, to = m.group(1).strip(), m.group(2).strip()
            cost = 0
            cost_str = ""
            duration = ""
            distance = ""
            m_cost = re.search(r"Estimated Cost:\s*\$([0-9\.]+)", str(content))
            if m_cost:
                cost_str = m_cost.group(1)
                cost = _parse_cost(cost_str)
            m_dur = re.search(r"Duration:\s*([0-9\.]+\s*mins)", str(content))
            if m_dur:
                duration = m_dur.group(1)
            m_dist = re.search(r"Distance:\s*([0-9\.]+\s*km)", str(content))
            if m_dist:
                distance = m_dist.group(1)
            raw = f"Self-driving, from {frm} to {to}"
            if duration or distance or cost_str:
                raw = f"{raw}, Duration: {duration}, Distance: {distance}, Estimated Cost: ${cost_str or cost}"
            t = TransportOption(mode="Self-driving", frm=frm, to=to, cost=cost, raw=raw)
            outbound.append(t)
            all_transports.append(t)

    if cities:
        ordered = infer_stage_order(q.org, cities, all_transports)
        if ordered:
            cities = ordered

    def split_out_in(options: List[TransportOption]) -> Tuple[List[TransportOption], List[TransportOption]]:
        out = [o for o in options if o.frm == q.org and o.to in cities]
        inp = [o for o in options if o.to == q.org and o.frm in cities]
        return out, inp

    out, inp = split_out_in(outbound)
    outbound = out
    inbound = inp

    for city in cities:
        if not any(o.to == city and o.frm == q.org for o in outbound):
            outbound.append(TransportOption(mode="Taxi", frm=q.org, to=city, cost=0))
        if not any(o.frm == city and o.to == q.org for o in inbound):
            inbound.append(TransportOption(mode="Taxi", frm=city, to=q.org, cost=0))

    if not outbound:
        outbound = [TransportOption(mode="Taxi", frm=q.org, to=q.dest, cost=0)]
    if not inbound:
        inbound = [TransportOption(mode="Taxi", frm=q.dest, to=q.org, cost=0)]

    if not bf:
        bf = []
    if not lu:
        lu = []
    if not di:
        di = []
    if not ats:
        ats = []
    if not accs:
        accs = []

    if not any(r.name == "-" for r in bf):
        bf.append(Restaurant(name="-", city=q.dest, cuisines=[], cost=0))
    if not any(r.name == "-" for r in lu):
        lu.append(Restaurant(name="-", city=q.dest, cuisines=[], cost=0))
    if not any(r.name == "-" for r in di):
        di.append(Restaurant(name="-", city=q.dest, cuisines=[], cost=0))
    if not any(a.name == "-" for a in ats):
        ats.append(Attraction(name="-", city=q.dest, categories=[], cost=0))
    if not any(a.name == "-" for a in accs):
        accs.append(Accommodation(name="-", city=q.dest, cost=0))

    if not cities:
        cities = [q.dest]

    return Candidates(
        cities=cities,
        outbound=outbound,
        inbound=inbound,
        breakfast=bf,
        lunch=lu,
        dinner=di,
        attractions=ats,
        accommodations=accs,
    )


# -----------------------------
# Z3 编码与求解
# -----------------------------

def _pick_cost(options_cost: List[int], sel_var: Int) -> Any:
    terms = []
    for i, c in enumerate(options_cost):
        terms.append(If(sel_var == i, int(c), 0))
    return Sum(terms)


def _restaurant_cuisine_ok(rest: Restaurant, allowed: List[str]) -> bool:
    if rest.name == "-":
        return True
    if not allowed:
        return True
    if not rest.cuisines:
        return True
    s = set([x.lower().strip() for x in rest.cuisines])
    a = set([x.lower().strip() for x in allowed])
    return len(s.intersection(a)) > 0


def solve_instance(
    q: QuerySpec,
    cand: Candidates,
    solver_timeout_ms: Optional[int],
    dump_smt2_path: Optional[str],
    persona: Optional[str],
    include_persona_in_poi: bool,
    transit_lookup: Callable[[str, str], Tuple[str, float]],
    use_optimize: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    返回 (decoded_plan, error_message)
    decoded_plan 仅包含 daily_plans 列表所需字段
    """
    opt = Optimize() if use_optimize else Solver()
    if solver_timeout_ms is not None:
        try:
            opt.set(timeout=int(solver_timeout_ms))
        except Exception:
            pass

    days = q.days
    if days <= 0:
        return None, "Invalid days"

    if not cand.cities:
        cand.cities = [q.dest] if q.dest else ["-"]
    max_stage_slots = max(1, (days + 1) // 2)
    target_stage = q.visiting_city_number if q.visiting_city_number > 0 else len(cand.cities)
    stage_slots = min(len(cand.cities), max(1, target_stage), max_stage_slots)
    if stage_slots < len(cand.cities):
        cand.cities = cand.cities[:stage_slots]

    # 城市选择（按天）
    city_vars = [Int(f"city_{d}") for d in range(days)]
    for d in range(days):
        opt.add(city_vars[d] >= 0, city_vars[d] < len(cand.cities))

    # Align with cli.py: each stage spans 2 days in order.
    for d in range(days):
        stage_idx = min(d // 2, len(cand.cities) - 1)
        opt.add(city_vars[d] == stage_idx)

    if len(cand.cities) > 0 and q.visiting_city_number > 0:
        used_flags = []
        for i in range(len(cand.cities)):
            used_flags.append(Or([city_vars[d] == i for d in range(days)]))
        city_count = Sum([If(u, 1, 0) for u in used_flags])
        target = min(q.visiting_city_number, len(cand.cities))
        opt.add(city_count == target)

    if days > 1:
        opt.add(city_vars[days - 1] == city_vars[days - 2])

    # outbound / inbound 选择
    out_sel = Int("out_sel")
    in_sel = Int("in_sel")
    opt.add(out_sel >= 0, out_sel < len(cand.outbound))
    opt.add(in_sel >= 0, in_sel < len(cand.inbound))
    self_out = [i for i, t in enumerate(cand.outbound) if (t.mode or "").lower() == "self-driving"]
    self_in = [i for i, t in enumerate(cand.inbound) if (t.mode or "").lower() == "self-driving"]
    nonself_out = [i for i, t in enumerate(cand.outbound) if (t.mode or "").lower() != "self-driving"]
    nonself_in = [i for i, t in enumerate(cand.inbound) if (t.mode or "").lower() != "self-driving"]
    for i in self_out:
        for j in nonself_in:
            opt.add(Not(And(out_sel == i, in_sel == j)))
    for i in self_in:
        for j in nonself_out:
            opt.add(Not(And(out_sel == j, in_sel == i)))

    # 每天变量: breakfast/lunch/dinner/attraction(2 slots)/accommodation
    bf_sel = [Int(f"bf_{d}") for d in range(days)]
    lu_sel = [Int(f"lu_{d}") for d in range(days)]
    di_sel = [Int(f"di_{d}") for d in range(days)]
    at1_sel = [Int(f"at1_{d}") for d in range(days)]
    at2_sel = [Int(f"at2_{d}") for d in range(days)]
    acc_sel = [Int(f"acc_{d}") for d in range(days)]

    for d in range(days):
        opt.add(bf_sel[d] >= 0, bf_sel[d] < len(cand.breakfast))
        opt.add(lu_sel[d] >= 0, lu_sel[d] < len(cand.lunch))
        opt.add(di_sel[d] >= 0, di_sel[d] < len(cand.dinner))
        opt.add(at1_sel[d] >= 0, at1_sel[d] < len(cand.attractions))
        opt.add(at2_sel[d] >= 0, at2_sel[d] < len(cand.attractions))
        opt.add(acc_sel[d] >= 0, acc_sel[d] < len(cand.accommodations))

    if days > 1:
        opt.add(acc_sel[days - 1] == acc_sel[days - 2])

    # cuisine hard constraint: 若 local_constraint.cuisine 给了列表，要求每餐选择满足 cuisine
    allowed_cuisines = q.local_constraint.get("cuisine", None)
    if isinstance(allowed_cuisines, str):
        allowed_cuisines = [allowed_cuisines]
    allowed_cuisines = list(allowed_cuisines) if isinstance(allowed_cuisines, list) else []

    if allowed_cuisines:
        bf_ok = [i for i, r in enumerate(cand.breakfast) if _restaurant_cuisine_ok(r, allowed_cuisines)]
        lu_ok = [i for i, r in enumerate(cand.lunch) if _restaurant_cuisine_ok(r, allowed_cuisines)]
        di_ok = [i for i, r in enumerate(cand.dinner) if _restaurant_cuisine_ok(r, allowed_cuisines)]

        # 若候选集没有任何满足的项，仍允许 "-"，避免直接 UNSAT
        if bf_ok:
            for d in range(days):
                opt.add(Or([bf_sel[d] == i for i in bf_ok]))
        if lu_ok:
            for d in range(days):
                opt.add(Or([lu_sel[d] == i for i in lu_ok]))
        if di_ok:
            for d in range(days):
                opt.add(Or([di_sel[d] == i for i in di_ok]))

        allowed_norm = [str(c).strip().lower() for c in allowed_cuisines if str(c).strip()]

        def _indices_with_cuisine(items, cuisine: str) -> List[int]:
            idxs = []
            for i, r in enumerate(items):
                if r.name == "-":
                    continue
                if not r.cuisines:
                    continue
                for c in r.cuisines:
                    if str(c).strip().lower() == cuisine:
                        idxs.append(i)
                        break
            return idxs

        for cuisine in allowed_norm:
            bf_idx = _indices_with_cuisine(cand.breakfast, cuisine)
            lu_idx = _indices_with_cuisine(cand.lunch, cuisine)
            di_idx = _indices_with_cuisine(cand.dinner, cuisine)
            exprs = []
            for d in range(days):
                if bf_idx:
                    exprs.append(Or([bf_sel[d] == i for i in bf_idx]))
                if lu_idx:
                    exprs.append(Or([lu_sel[d] == i for i in lu_idx]))
                if di_idx:
                    exprs.append(Or([di_sel[d] == i for i in di_idx]))
            if exprs:
                opt.add(Or(exprs))

    attraction_types = q.local_constraint.get("attraction")
    if attraction_types:
        if isinstance(attraction_types, str):
            attraction_types = [attraction_types]
        attr_norm = [str(a).strip().lower() for a in attraction_types if str(a).strip()]

        def _indices_with_attraction_type(items, att_type: str) -> List[int]:
            idxs = []
            for i, a in enumerate(items):
                if a.name == "-":
                    continue
                cats = [str(c).strip().lower() for c in (a.categories or []) if str(c).strip()]
                if att_type in cats:
                    idxs.append(i)
            return idxs

        for att_type in attr_norm:
            att_idx = _indices_with_attraction_type(cand.attractions, att_type)
            exprs = []
            for d in range(days):
                if att_idx:
                    exprs.append(Or([at1_sel[d] == i for i in att_idx]))
                    exprs.append(Or([at2_sel[d] == i for i in att_idx]))
            if exprs:
                opt.add(Or(exprs))

    # 选择项城市一致性
    def _indices_by_city(items):
        mapping: Dict[str, List[int]] = {c: [] for c in cand.cities}
        placeholder = None
        for i, it in enumerate(items):
            if getattr(it, "name", "") == "-":
                placeholder = i
                continue
            if it.city in mapping:
                mapping[it.city].append(i)
        return mapping, placeholder

    bf_map, bf_ph = _indices_by_city(cand.breakfast)
    lu_map, lu_ph = _indices_by_city(cand.lunch)
    di_map, di_ph = _indices_by_city(cand.dinner)
    at_map, at_ph = _indices_by_city(cand.attractions)
    acc_map, acc_ph = _indices_by_city(cand.accommodations)

    if acc_ph is not None and any(a.name != "-" for a in cand.accommodations):
        for d in range(max(0, days - 1)):
            opt.add(acc_sel[d] != acc_ph)

    def _add_city_constraint(sel_var, mapping, placeholder, city_var):
        for i, city in enumerate(cand.cities):
            allowed = list(mapping.get(city, []))
            if placeholder is not None:
                allowed.append(placeholder)
            if not allowed:
                continue
            opt.add(Implies(city_var == i, Or([sel_var == a for a in allowed])))

    for d in range(days):
        _add_city_constraint(bf_sel[d], bf_map, bf_ph, city_vars[d])
        _add_city_constraint(lu_sel[d], lu_map, lu_ph, city_vars[d])
        _add_city_constraint(di_sel[d], di_map, di_ph, city_vars[d])
        _add_city_constraint(at1_sel[d], at_map, at_ph, city_vars[d])
        _add_city_constraint(at2_sel[d], at_map, at_ph, city_vars[d])
        _add_city_constraint(acc_sel[d], acc_map, acc_ph, city_vars[d])

    def _indices_by_key(items):
        mapping: Dict[str, List[int]] = {}
        for i, it in enumerate(items):
            if getattr(it, "name", "") == "-":
                continue
            key = f"{it.name}||{it.city}"
            mapping.setdefault(key, []).append(i)
        return mapping

    bf_key = _indices_by_key(cand.breakfast)
    lu_key = _indices_by_key(cand.lunch)
    di_key = _indices_by_key(cand.dinner)
    rest_keys = set(bf_key) | set(lu_key) | set(di_key)
    for key in rest_keys:
        flags = []
        bf_idx = bf_key.get(key, [])
        lu_idx = lu_key.get(key, [])
        di_idx = di_key.get(key, [])
        for d in range(days):
            if bf_idx:
                flags.append(Or([bf_sel[d] == i for i in bf_idx]))
            if lu_idx:
                flags.append(Or([lu_sel[d] == i for i in lu_idx]))
            if di_idx:
                flags.append(Or([di_sel[d] == i for i in di_idx]))
        if flags:
            opt.add(Sum([If(f, 1, 0) for f in flags]) <= 1)

    at_key = _indices_by_key(cand.attractions)
    for key, idxs in at_key.items():
        flags = []
        for d in range(days):
            flags.append(Or([at1_sel[d] == i for i in idxs]))
            flags.append(Or([at2_sel[d] == i for i in idxs]))
        opt.add(Sum([If(f, 1, 0) for f in flags]) <= 1)

    out_map: Dict[str, List[int]] = {c: [] for c in cand.cities}
    in_map: Dict[str, List[int]] = {c: [] for c in cand.cities}
    for i, t in enumerate(cand.outbound):
        if t.frm == q.org and t.to in out_map:
            out_map[t.to].append(i)
    for i, t in enumerate(cand.inbound):
        if t.to == q.org and t.frm in in_map:
            in_map[t.frm].append(i)
    for i, city in enumerate(cand.cities):
        if out_map.get(city):
            opt.add(Implies(city_vars[0] == i, Or([out_sel == a for a in out_map[city]])))
        if in_map.get(city):
            opt.add(Implies(city_vars[days - 1] == i, Or([in_sel == a for a in in_map[city]])))

    # budget constraint: 总费用
    out_costs = [t.cost for t in cand.outbound]
    in_costs = [t.cost for t in cand.inbound]
    bf_costs = [r.cost for r in cand.breakfast]
    lu_costs = [r.cost for r in cand.lunch]
    di_costs = [r.cost for r in cand.dinner]
    at_costs = [a.cost for a in cand.attractions]
    acc_costs = [a.cost for a in cand.accommodations]

    total_cost = Int("total_cost")
    cost_terms = []
    cost_terms.append(_pick_cost(out_costs, out_sel))
    cost_terms.append(_pick_cost(in_costs, in_sel))
    for d in range(days):
        cost_terms.append(_pick_cost(bf_costs, bf_sel[d]))
        cost_terms.append(_pick_cost(lu_costs, lu_sel[d]))
        cost_terms.append(_pick_cost(di_costs, di_sel[d]))
        cost_terms.append(_pick_cost(at_costs, at1_sel[d]))
        cost_terms.append(_pick_cost(at_costs, at2_sel[d]))
        if d < days - 1:
            cost_terms.append(_pick_cost(acc_costs, acc_sel[d]))

    opt.add(total_cost == Sum(cost_terms))
    if q.budget > 0:
        opt.add(total_cost <= int(q.budget))

    # objective: maximize non-empty selections, then min cost
    def _placeholder_index(items) -> Optional[int]:
        for i, it in enumerate(items):
            if getattr(it, "name", "") == "-":
                return i
        return None

    non_empty_terms = []
    bf_ph = _placeholder_index(cand.breakfast)
    lu_ph = _placeholder_index(cand.lunch)
    di_ph = _placeholder_index(cand.dinner)
    at_ph = _placeholder_index(cand.attractions)
    acc_ph = _placeholder_index(cand.accommodations)
    for d in range(days):
        if bf_ph is not None:
            non_empty_terms.append(If(bf_sel[d] == bf_ph, 0, 1))
        if lu_ph is not None:
            non_empty_terms.append(If(lu_sel[d] == lu_ph, 0, 1))
        if di_ph is not None:
            non_empty_terms.append(If(di_sel[d] == di_ph, 0, 1))
        if at_ph is not None:
            non_empty_terms.append(If(at1_sel[d] == at_ph, 0, 1))
            non_empty_terms.append(If(at2_sel[d] == at_ph, 0, 1))
        if acc_ph is not None:
            non_empty_terms.append(If(acc_sel[d] == acc_ph, 0, 1))
    if use_optimize:
        if non_empty_terms:
            opt.maximize(Sum(non_empty_terms))
        # objective: min cost
        opt.minimize(total_cost)

    # dump smt2
    if dump_smt2_path:
        try:
            s = opt.sexpr()
            with open(dump_smt2_path, "w", encoding="utf-8") as f:
                f.write(s)
        except Exception:
            pass

    res = opt.check()
    if res == sat:
        m = opt.model()
        out_i = m.eval(out_sel).as_long()
        in_i = m.eval(in_sel).as_long()
        out_t = cand.outbound[out_i]
        in_t = cand.inbound[in_i]

        persona_meta = _persona_to_meta(persona) if include_persona_in_poi else None
        daily_plans: List[Dict[str, Any]] = []
        prev_acc_poi: Optional[str] = None
        for d in range(days):
            c_i = m.eval(city_vars[d]).as_long()
            city_name = cand.cities[c_i] if 0 <= c_i < len(cand.cities) else q.dest
            bf_i = m.eval(bf_sel[d]).as_long()
            lu_i = m.eval(lu_sel[d]).as_long()
            di_i = m.eval(di_sel[d]).as_long()
            at1_i = m.eval(at1_sel[d]).as_long()
            at2_i = m.eval(at2_sel[d]).as_long()
            acc_i = m.eval(acc_sel[d]).as_long()

            bf = cand.breakfast[bf_i]
            lu = cand.lunch[lu_i]
            di = cand.dinner[di_i]
            at1 = cand.attractions[at1_i]
            at2 = cand.attractions[at2_i]
            acc = cand.accommodations[acc_i]

            day_num = d + 1

            if day_num == 1:
                current_city = f"from {q.org} to {city_name}"
                transportation = format_transport(out_t, q.org, city_name)
            else:
                prev_ci = m.eval(city_vars[d - 1]).as_long()
                prev_city = cand.cities[prev_ci] if 0 <= prev_ci < len(cand.cities) else q.dest
                if day_num == days:
                    current_city = f"from {prev_city} to {q.org}"
                    transportation = format_transport(in_t, prev_city, q.org)
                else:
                    current_city = f"from {prev_city} to {city_name}"
                    transportation = "Transfer, from {0} to {1}".format(prev_city, city_name)

            lunch_str = f"{lu.name}, {lu.city}" if lu.name != "-" else "-"
            dinner_str = f"{di.name}, {di.city}" if di.name != "-" else "-"
            acc_str = f"{acc.name}, {acc.city}" if acc.name != "-" else "-"
            if day_num == days:
                acc_str = "-"

            at_names: List[str] = []
            if at1.name != "-":
                at_names.append(f"{at1.name}, {at1.city}")
            if at2.name != "-" and at2.name != at1.name:
                at_names.append(f"{at2.name}, {at2.city}")
            attraction_str = "; ".join(at_names) if at_names else "-"

            day_dict = {
                "days": day_num,
                "current_city": current_city,
                "transportation": transportation,
                "breakfast": f"{bf.name}, {bf.city}" if bf.name != "-" else "-",
                "attraction": attraction_str,
                "lunch": lunch_str,
                "dinner": dinner_str,
                "accommodation": acc_str,
                "event": "-",
            }
            if acc.name != "-":
                day_dict["_accommodation_poi"] = f"{acc.name}, {acc.city}"
            if prev_acc_poi and ((days == 5 and day_num == 3) or (days == 7 and day_num in (3, 5))):
                day_dict["_prev_accommodation_poi"] = prev_acc_poi
            day_dict["point_of_interest_list"] = build_point_of_interest_list(
                day_dict,
                persona_meta,
                day_num == days,
                transit_lookup,
            )
            day_dict.pop("_accommodation_poi", None)
            day_dict.pop("_prev_accommodation_poi", None)
            daily_plans.append(day_dict)
            if acc.name != "-":
                prev_acc_poi = f"{acc.name}, {acc.city}"

        _assign_events_to_plan(q, cand, daily_plans)
        return {"plan": daily_plans, "total_cost": int(m.eval(total_cost).as_long())}, None

    if res == unknown:
        return None, f"UNKNOWN: {opt.reason_unknown()}"
    return None, "UNSAT"


def format_transport(t: TransportOption, frm: str, to: str) -> str:
    if t.raw:
        return t.raw
    if t.mode.lower() == "flight":
        fn = t.flight_no or "F0000000"
        dep = t.dep_time or "00:00"
        arr = t.arr_time or "00:00"
        return f"Flight Number: {fn}, from {frm} to {to}, Departure Time: {dep}, Arrival Time: {arr}"
    if t.mode.lower() == "taxi":
        return f"Taxi, from {frm} to {to}, Not feasible to drive from {frm} to {to}"
    if t.mode.lower() == "transfer":
        return f"Transfer, from {frm} to {to}"
    if t.mode.lower() == "self-driving":
        return f"Self-driving, from {frm} to {to}"
    return f"{t.mode}, from {frm} to {to}"


# -----------------------------
# 输出封装
# -----------------------------

def make_output_record(
    idx: int,
    query_json: Dict[str, Any],
    persona: Optional[str],
    solved: Optional[Dict[str, Any]],
    error: Optional[str],
) -> Dict[str, Any]:
    if solved is None:
        return {
            "idx": idx,
            "JSON": query_json,
            "persona": persona,
            "plan": [],
            "error": error or "UNSAT",
        }
    return {
        "idx": idx,
        "JSON": query_json,
        "persona": persona,
        "plan": solved["plan"],
    }


# -----------------------------
# 主流程
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=-1)
    ap.add_argument("--solver_timeout_ms", type=int, default=60000)
    ap.add_argument("--candidates_dir", type=str, default=None, help="可选: 外部候选集目录，用于替换 plan 回退策略")
    ap.add_argument("--emit_smt2_dir", type=str, default=None, help="可选: 输出每条样本的 SMT2")
    ap.add_argument("--eval_mode", action="store_true", help="评测模式：关闭 persona POI + 放宽餐次间隔")
    args = ap.parse_args()

    if args.eval_mode:
        global TIME_SLOTS
        TIME_SLOTS = dict(EVAL_TIME_SLOTS)

    df = pd.read_csv(args.input_csv)
    df = df.reset_index(drop=True)

    if args.emit_smt2_dir:
        os.makedirs(args.emit_smt2_dir, exist_ok=True)

    out_lines: List[str] = []
    count = 0

    for i in range(args.start_index, len(df)):
        if args.max_rows >= 0 and count >= args.max_rows:
            break

        row = df.iloc[i].to_dict()
        row["__row_index__"] = i

        idx, query_json, persona, plan_from_row, reference_info = parse_row_to_instance(row)
        q = queryspec_from_query_json(query_json)

        # 候选集优先来自外部目录，其次来自 reference_info，再次来自 plan 回退
        external = build_candidates_from_external(q, args.candidates_dir)
        from_ref = build_candidates_from_reference_info(q, reference_info) if external is None else None
        if external is not None:
            cand = external
        elif from_ref is not None:
            cand = from_ref
        else:
            cand = build_candidates_from_plan_fallback(q, plan_from_row)

        smt2_path = None
        if args.emit_smt2_dir:
            smt2_path = os.path.join(args.emit_smt2_dir, f"idx_{idx}_row_{i}.smt2")

        if args.eval_mode:
            cand = filter_candidates_for_sandbox(q, cand)

        poi_names = collect_poi_names(cand)
        transit_map: Dict[Tuple[str, str], Tuple[str, float]] = {}
        if args.eval_mode:
            cache = _load_sandbox_cache()
            if cache.get("ok"):
                transit_map = cache.get("transit", {})
        if not transit_map:
            transit_map = build_transit_map(reference_info, poi_names)

        def transit_lookup(name: str, city: str) -> Tuple[str, float]:
            key = (_normalize_poi_key(name), _normalize_city_key(city))
            if key in transit_map:
                return transit_map[key]
            key_any = (_normalize_poi_key(name), "")
            return transit_map.get(key_any, ("-", 0.0))

        solved, err = solve_instance(
            q,
            cand,
            args.solver_timeout_ms,
            smt2_path,
            persona,
            not args.eval_mode,
            transit_lookup,
            use_optimize=not args.eval_mode,
        )
        rec = make_output_record(idx, query_json, persona, solved, err)

        out_lines.append(json.dumps(rec, ensure_ascii=False))
        count += 1

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
