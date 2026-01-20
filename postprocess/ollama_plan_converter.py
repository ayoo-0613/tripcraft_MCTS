import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional


PROMPT_PREFIX = """Extract a travel plan from the text and return a JSON array. Each array item must include exactly these keys:
"days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
Use "-" if any field is missing. Keep names and punctuation exactly as in the text. For multiple attractions, use ';' to separate.
Return only valid JSON (double quotes, no trailing commas).

Example:
Text:
Day 1:
Current City: from A to B
Transportation: Flight Number: F123, from A to B, Departure Time: 09:00, Arrival Time: 10:00
Breakfast: -
Attraction: Museum X, B; Park Y, B
Lunch: Cafe Z, B
Dinner: -
Accommodation: Hotel Q, B
Event: -
Point of Interest List: Hotel Q, stay from 10:30 to 11:00; Museum X, visit from 12:00 to 13:00.

JSON:
[
  {
    "days": 1,
    "current_city": "from A to B",
    "transportation": "Flight Number: F123, from A to B, Departure Time: 09:00, Arrival Time: 10:00",
    "breakfast": "-",
    "attraction": "Museum X, B; Park Y, B",
    "lunch": "Cafe Z, B",
    "dinner": "-",
    "accommodation": "Hotel Q, B",
    "event": "-",
    "point_of_interest_list": "Hotel Q, stay from 10:30 to 11:00; Museum X, visit from 12:00 to 13:00."
  }
]

Text:
"""


def _load_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _parse_literal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return value
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _build_json_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "org": row.get("org"),
        "dest": row.get("dest"),
        "days": int(row.get("days") or 0),
        "visiting_city_number": int(row.get("visiting_city_number") or 0),
        "date": _parse_literal(row.get("date") or []),
        "people_number": int(row.get("people_number") or 0),
        "local_constraint": _parse_literal(row.get("local_constraint") or {}),
        "budget": float(row.get("budget") or 0),
        "query": row.get("query") or None,
        "level": row.get("level") or None,
    }


def _detect_plan_key(record: Dict[str, Any]) -> Optional[str]:
    for key in record:
        if key.endswith("_sole-planning_results"):
            return key
    return None


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    m = re.search(r"\\[.*\\]", text, flags=re.DOTALL)
    if m:
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(chunk)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return None


def _default_plan() -> List[Dict[str, Any]]:
    return [
        {
            "days": 0,
            "current_city": "-",
            "transportation": "-",
            "breakfast": "-",
            "attraction": "-",
            "lunch": "-",
            "dinner": "-",
            "accommodation": "-",
            "event": "-",
            "point_of_interest_list": "-",
        }
    ]


def _ollama_chat(base_url: str, model: str, prompt: str, timeout: float) -> str:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("message") or {}).get("content")
    return str(content or "")


def _convert_plan_text(
    plan_text: str,
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> List[Dict[str, Any]]:
    prompt = PROMPT_PREFIX + plan_text.strip() + "\n\nJSON:\n"
    for attempt in range(max_retries + 1):
        try:
            response = _ollama_chat(base_url, model, prompt, timeout)
            parsed = _extract_json_array(response)
            if parsed is not None:
                return parsed
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(1)
    return _default_plan()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--plan_key", type=str, default="")
    parser.add_argument("--ollama_model", type=str, default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    parser.add_argument("--ollama_base_url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--ollama_timeout", type=float, default=float(os.environ.get("OLLAMA_TIMEOUT", "999")))
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=1)
    args = parser.parse_args()

    rows = _load_csv_rows(args.csv_file)

    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fixed_plan_key = args.plan_key.strip() or None
    max_samples = args.max_samples

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line_idx, line in enumerate(fin, start=1):
            if max_samples and line_idx > max_samples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            idx = int(record.get("index") or line_idx)
            if idx < 1 or idx > len(rows):
                print(f"Skip idx {idx}: out of range for CSV rows.", file=sys.stderr)
                continue

            row = rows[idx - 1]
            json_payload = _build_json_payload(row)
            persona = row.get("persona") or ""

            plan_key = fixed_plan_key or _detect_plan_key(record)
            plan_text = record.get(plan_key) if plan_key else None
            if not plan_text:
                print(f"Skip idx {idx}: plan text missing.", file=sys.stderr)
                plan = _default_plan()
            else:
                plan = _convert_plan_text(
                    plan_text=plan_text,
                    base_url=args.ollama_base_url,
                    model=args.ollama_model,
                    timeout=args.ollama_timeout,
                    max_retries=args.max_retries,
                )

            out_record = {
                "idx": idx,
                "JSON": json_payload,
                "persona": persona,
                "plan": plan,
            }
            json.dump(out_record, fout, ensure_ascii=True)
            fout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
