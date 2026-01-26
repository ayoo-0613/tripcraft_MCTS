import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from langchain.prompts import PromptTemplate
from tools.planner.apis import Planner

from mcts_baseline.io import load_rows, TripCraftRow
from mcts_baseline.templater import make_output_record_template
from mcts_baseline.ref_parser import build_unified_kb
from mcts_baseline.env import TripCraftEnv
from mcts_baseline.formatter import fill_template_with_state, _stage_city_for_day, _with_city


PROMPT_DIRECT = """You are a planner. Assign actions for each day based ONLY on the given information.
Output ONLY valid JSON. Do NOT include explanations.

Rules:
- Use real names from the given information.
- Use "-" for any slot you want to skip.
- Use at most 2 attractions per day.
- Output must exactly match the schema below.

Schema:
{{
  "days": [
    {{
      "day": 1,
      "transportation": "Flight Number: ...",
      "accommodation": "Hotel Name, City",
      "breakfast": "Restaurant Name, City",
      "lunch": "Restaurant Name, City",
      "dinner": "Restaurant Name, City",
      "attractions": ["Attraction A, City", "Attraction B, City"],
      "event": "-"
    }}
  ]
}}

Given information:
{text}
Query: {query}
Traveler Persona:
{persona}
"""

PROMPT_COT = """You are a planner. Assign actions for each day based ONLY on the given information.
Think step-by-step internally, but output ONLY the JSON object described below.

Schema:
{{
  "days": [
    {{
      "day": 1,
      "transportation": "Flight Number: ...",
      "accommodation": "Hotel Name, City",
      "breakfast": "Restaurant Name, City",
      "lunch": "Restaurant Name, City",
      "dinner": "Restaurant Name, City",
      "attractions": ["Attraction A, City", "Attraction B, City"],
      "event": "-"
    }}
  ]
}}

Given information:
{text}
Query: {query}
Traveler Persona:
{persona}
"""

PROMPT_REACT = """You are a planner. Use a ReAct-style internal process, but output ONLY the JSON object.
Assign actions for each day based ONLY on the given information.

Schema:
{{
  "days": [
    {{
      "day": 1,
      "transportation": "Flight Number: ...",
      "accommodation": "Hotel Name, City",
      "breakfast": "Restaurant Name, City",
      "lunch": "Restaurant Name, City",
      "dinner": "Restaurant Name, City",
      "attractions": ["Attraction A, City", "Attraction B, City"],
      "event": "-"
    }}
  ]
}}

Given information:
{text}
Query: {query}
Traveler Persona:
{persona}
"""

PROMPT_REFLEXION = """You are a planner. Use a Reflexion-style internal process, but output ONLY the JSON object.
Assign actions for each day based ONLY on the given information.

Schema:
{{
  "days": [
    {{
      "day": 1,
      "transportation": "Flight Number: ...",
      "accommodation": "Hotel Name, City",
      "breakfast": "Restaurant Name, City",
      "lunch": "Restaurant Name, City",
      "dinner": "Restaurant Name, City",
      "attractions": ["Attraction A, City", "Attraction B, City"],
      "event": "-"
    }}
  ]
}}

Given information:
{text}
Query: {query}
Traveler Persona:
{persona}
"""


PROMPT_BY_STRATEGY = {
    "direct": PROMPT_DIRECT,
    "cot": PROMPT_COT,
    "react": PROMPT_REACT,
    "reflexion": PROMPT_REFLEXION,
}


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _resolve_day_plan(day_choice: Dict[str, Any], env: TripCraftEnv, day: int) -> Dict[str, Any]:
    transport = str(day_choice.get("transportation") or "-")
    accommodation = str(day_choice.get("accommodation") or "-")
    breakfast = str(day_choice.get("breakfast") or "-")
    lunch = str(day_choice.get("lunch") or "-")
    dinner = str(day_choice.get("dinner") or "-")
    event = str(day_choice.get("event") or "-")

    attractions_raw = day_choice.get("attractions") or []
    if isinstance(attractions_raw, str):
        attractions = [attractions_raw]
    elif isinstance(attractions_raw, list):
        attractions = [str(x) for x in attractions_raw if x]
    else:
        attractions = []
    attractions = [a for a in attractions if a != "-"][:2]

    _, frm, to = env._city_movement_for_day(day)
    return {
        "transportation": transport,
        "accommodation": accommodation,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner,
        "attractions": attractions,
        "event": event,
        "from": frm,
        "to": to,
    }


def _apply_day_plan(env: TripCraftEnv, state, day_plan: Dict[str, Any]) -> None:
    while True:
        slot = env._slot_name(state)
        if slot == "transport":
            action = {
                "type": "set_transport",
                "raw": day_plan.get("transportation", "-"),
                "from": day_plan.get("from") or "",
                "to": day_plan.get("to") or "",
            }
        elif slot == "accommodation":
            name = day_plan.get("accommodation", "-")
            if name == "-":
                action = {"type": "skip_accommodation", "name": "-"}
            else:
                action = {"type": "set_accommodation", "name": name}
        elif slot == "breakfast":
            name = day_plan.get("breakfast", "-")
            if name == "-":
                action = {"type": "skip_breakfast", "name": "-"}
            else:
                action = {"type": "set_breakfast", "name": name}
        elif slot == "attraction1":
            name = (day_plan.get("attractions") or ["-"])[0]
            if not name or name == "-":
                action = {"type": "skip_attraction1", "name": "-"}
            else:
                action = {"type": "add_attraction1", "name": name}
        elif slot == "lunch":
            name = day_plan.get("lunch", "-")
            if name == "-":
                action = {"type": "skip_lunch", "name": "-"}
            else:
                action = {"type": "set_lunch", "name": name}
        elif slot == "attraction2":
            name = (day_plan.get("attractions") or ["-", "-"])[1]
            if not name or name == "-":
                action = {"type": "skip_attraction2", "name": "-"}
            else:
                action = {"type": "add_attraction2", "name": name}
        elif slot == "dinner":
            name = day_plan.get("dinner", "-")
            if name == "-":
                action = {"type": "skip_dinner", "name": "-"}
            else:
                action = {"type": "set_dinner", "name": name}
        elif slot == "end_day":
            action = {"type": "end_day", "name": "-"}
        else:
            action = {"type": f"skip_{slot}", "name": "-"}

        env.apply_action(state, action, record_trace=False)
        if slot == "end_day":
            break


def _collect_reference_text(row: TripCraftRow) -> str:
    blocks = [b for b in (row.ref_blocks or []) if b]
    return "\n\n".join(str(b) for b in blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--strategy", type=str, default="direct", choices=["direct", "cot", "react", "reflexion"])
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    args = parser.parse_args()

    csv_path = args.csv_file
    if not os.path.isabs(csv_path):
        repo_relative = os.path.join(_ROOT, csv_path)
        if os.path.exists(repo_relative):
            csv_path = repo_relative
    rows = load_rows(csv_path)

    prompt_template = PromptTemplate(
        input_variables=["text", "query", "persona"],
        template=PROMPT_BY_STRATEGY[args.strategy],
    )
    planner = Planner(model_name=args.model_name, agent_prompt=prompt_template)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8", errors="replace") as f:
        for row in rows:
            template = make_output_record_template(row)
            kb = build_unified_kb(row.org, row.ref_blocks)
            env = TripCraftEnv(row=row, kb=kb, topk=5)
            state = env.initial_state()

            text = _collect_reference_text(row)
            llm_raw = planner.run(text, row.query or "", row.persona or "")
            choice = _safe_json_loads(llm_raw) or {"days": []}

            day_choices = {int(d.get("day")): d for d in choice.get("days", []) if isinstance(d, dict)}
            day_plans: Dict[int, Dict[str, Any]] = {}
            for day in range(1, row.days + 1):
                day_choice = day_choices.get(day, {})
                day_plans[day] = _resolve_day_plan(day_choice, env, day)
                _apply_day_plan(env, state, day_plans[day])

            record = fill_template_with_state(template, row, kb, state)

            for day in range(1, row.days + 1):
                event = day_plans.get(day, {}).get("event", "-")
                city = _stage_city_for_day(kb, day)
                record["plan"][day - 1]["event"] = _with_city(event, city) if event and event != "-" else "-"

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.sleep_sec:
                time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
