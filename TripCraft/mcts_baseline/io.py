from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TripCraftRow:
    idx: int
    org: str
    dest: str
    days: int
    visiting_city_number: int
    date: List[str]
    people_number: int
    local_constraint: Dict[str, Any]
    budget: float
    query: Optional[str]
    level: str
    persona: str
    ref_blocks: List[str]


def _maybe_literal_eval(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _normalize_ref_blocks(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return [json.dumps([value], ensure_ascii=True)]
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, dict) for item in value):
            return [json.dumps(value, ensure_ascii=True)]
        blocks: List[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    blocks.append(text)
            elif isinstance(item, dict):
                blocks.append(json.dumps([item], ensure_ascii=True))
            else:
                blocks.append(str(item))
        return blocks
    return [str(value)]


def row_from_dict(data: Dict[str, Any], idx_default: int = 1) -> TripCraftRow:
    payload: Dict[str, Any] = {}
    json_payload = data.get("JSON")
    if isinstance(json_payload, dict):
        payload.update(json_payload)
    payload.update({k: v for k, v in data.items() if k not in {"plan", "JSON"}})

    idx = int(payload.get("idx") or idx_default)
    days = int(payload.get("days") or 0)

    date_val = _maybe_literal_eval(payload.get("date"))
    if isinstance(date_val, list):
        dates = [str(x) for x in date_val]
    else:
        dates = [str(date_val)] if date_val is not None else []

    local_constraint = _maybe_literal_eval(payload.get("local_constraint"))
    if not isinstance(local_constraint, dict):
        local_constraint = {}

    budget_raw = payload.get("budget")
    try:
        budget = float(budget_raw) if budget_raw is not None else 0.0
    except Exception:
        budget = 0.0

    query = payload.get("query")
    if isinstance(query, str) and query.strip() == "":
        query = None

    ref_blocks = _normalize_ref_blocks(payload.get("ref_blocks"))
    if not ref_blocks:
        ref_blocks = _normalize_ref_blocks(payload.get("reference_information"))
    if not ref_blocks:
        for k in (1, 2, 3):
            ref_blocks.extend(_normalize_ref_blocks(payload.get(f"reference_information_{k}")))

    return TripCraftRow(
        idx=idx,
        org=str(payload.get("org") or ""),
        dest=str(payload.get("dest") or ""),
        days=days,
        visiting_city_number=int(payload.get("visiting_city_number") or 0),
        date=dates,
        people_number=int(payload.get("people_number") or 0),
        local_constraint=local_constraint,
        budget=budget,
        query=query,
        level=str(payload.get("level") or ""),
        persona=str(payload.get("persona") or ""),
        ref_blocks=ref_blocks,
    )


def load_rows_from_json(json_path: str) -> List[TripCraftRow]:
    text = Path(json_path).read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped == "":
        return []
    items: List[Dict[str, Any]] = []
    if stripped[0] in "[{":
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                items = [obj]
            elif isinstance(obj, list):
                items = [x for x in obj if isinstance(x, dict)]
        except Exception:
            items = []
    if not items:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return [row_from_dict(item, idx_default=i + 1) for i, item in enumerate(items)]


def load_rows(csv_path: str) -> List[TripCraftRow]:
    """
    Read CSV and normalize reference columns:
      - if 'reference_information' exists -> ref_blocks=[...]
      - else collect 'reference_information_1..3' into ref_blocks
    Parse 'date' and 'local_constraint' with ast.literal_eval.
    idx: if not provided, use row number (starting from 1).
    """
    rows: List[TripCraftRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, raw in enumerate(reader, start=1):
            idx = int(raw.get("idx") or row_num)

            days = int(raw["days"])
            date_val = _maybe_literal_eval(raw.get("date"))
            if isinstance(date_val, list):
                dates = [str(x) for x in date_val]
            else:
                dates = [str(date_val)] if date_val is not None else []

            local_constraint = _maybe_literal_eval(raw.get("local_constraint"))
            if not isinstance(local_constraint, dict):
                local_constraint = {}

            budget_raw = raw.get("budget")
            try:
                budget = float(budget_raw) if budget_raw is not None else 0.0
            except Exception:
                budget = 0.0

            query = raw.get("query")
            if query is not None and query.strip() == "":
                query = None

            if "reference_information" in raw and raw.get("reference_information"):
                ref_blocks = [raw["reference_information"]]
            else:
                ref_blocks = []
                for k in (1, 2, 3):
                    col = f"reference_information_{k}"
                    if col in raw and raw.get(col):
                        ref_blocks.append(raw[col])

            rows.append(
                TripCraftRow(
                    idx=idx,
                    org=str(raw.get("org") or ""),
                    dest=str(raw.get("dest") or ""),
                    days=days,
                    visiting_city_number=int(raw.get("visiting_city_number") or 0),
                    date=dates,
                    people_number=int(raw.get("people_number") or 0),
                    local_constraint=local_constraint,
                    budget=budget,
                    query=query,
                    level=str(raw.get("level") or ""),
                    persona=str(raw.get("persona") or ""),
                    ref_blocks=ref_blocks,
                )
            )
    return rows
