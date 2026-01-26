import sys
import os

# Add repo root to sys.path (robust to cwd)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# Now you can safely import the necessary modules
import re
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
from tools.planner.apis import Planner
from mcts_baseline.io import row_from_dict
from mcts_baseline.templater import make_output_record_template
from mcts_baseline.ref_parser import build_unified_kb
from mcts_baseline.env import TripCraftEnv
from mcts_baseline.formatter import fill_template_with_state, _stage_city_for_day, _with_city
from mcts_baseline.retrieval import topk_restaurants, topk_accommodations, topk_attractions
import openai

# Change the working directory if needed
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from agents.prompts import (
    planner_agent_prompt_direct_og,
    planner_agent_prompt_direct_param,
    planner_agent_prompt_llm_direct,
    planner_agent_prompt_llm_direct_repair,
    planner_agent_prompt_llm_cot_plan,
    planner_agent_prompt_llm_cot_fill,
    planner_agent_prompt_llm_reflexion_repair,
)


def _safe_json_loads(text: str):
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


def _resolve_day_plan(day_choice: dict, env: TripCraftEnv, day: int) -> dict:
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


def _unique_keep(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _transport_candidates(kb, frm: str, to: str, date: str, k: int) -> list:
    if not kb.transports:
        return ["-"]
    candidates = [t for t in kb.transports if t.frm == frm and t.to == to and (t.date is None or t.date == date)]
    if not candidates:
        candidates = [t for t in kb.transports if t.frm == frm and t.to == to]

    def _rank(t) -> int:
        mode_rank = {"flight": 3, "self-driving": 2, "taxi": 1}.get(t.mode, 0)
        has_detail = 1 if ("Flight Number:" in (t.raw or "")) else 0
        return mode_rank * 10 + has_detail

    candidates.sort(key=_rank, reverse=True)
    raws = [t.raw or f"{t.mode.title()} from {frm} to {to}" for t in candidates]
    raws = _unique_keep([r for r in raws if r])
    if not raws:
        return ["-"]
    return raws[:k]


def _stage_candidates(row, kb, env: TripCraftEnv, day: int, topk: int) -> dict:
    stage = env._stage_for_day(day)
    current_city, frm, to = env._city_movement_for_day(day)
    travel = env._day_is_travel(day)

    date = None
    if row.date and 0 <= (day - 1) < len(row.date):
        date = row.date[day - 1]

    transport_list = _transport_candidates(kb, frm, to, date, topk) if travel else ["-"]

    meals = {"breakfast": [], "lunch": [], "dinner": []}
    attractions = []
    accommodations = []
    events = []

    if stage is not None:
        for meal in meals:
            meals[meal] = [c["name"] for c in topk_restaurants(stage, meal, row.local_constraint or {}, topk)]
        attractions = [c["name"] for c in topk_attractions(stage, row.local_constraint or {}, topk * 2)]
        accommodations = [c["name"] for c in topk_accommodations(stage, row.local_constraint or {}, topk)]
        if getattr(stage, "events", None) is not None and not stage.events.empty and "name" in stage.events.columns:
            events = [str(x) for x in stage.events["name"].tolist() if x]

    def _add_skip(items):
        cleaned = [x for x in items if x]
        cleaned.append("-")
        return _unique_keep(cleaned)

    return {
        "day": day,
        "current_city": current_city,
        "from": frm,
        "to": to,
        "transportation": _add_skip(transport_list),
        "accommodation": _add_skip([] if day == row.days else accommodations),
        "breakfast": _add_skip(meals["breakfast"]),
        "lunch": _add_skip(meals["lunch"]),
        "dinner": _add_skip(meals["dinner"]),
        "attractions": _add_skip(attractions),
        "events": _add_skip(events),
    }


def _build_candidates_payload(row, kb, env: TripCraftEnv, topk: int) -> dict:
    days = [_stage_candidates(row, kb, env, d, topk) for d in range(1, row.days + 1)]
    return {"days": days}


def _format_candidates_for_llm(candidates_payload: dict) -> str:
    def _numbered(items):
        return [{"idx": i, "value": v} for i, v in enumerate(items)]

    formatted = {"days": []}
    for day in candidates_payload.get("days", []):
        formatted["days"].append(
            {
                "day": day.get("day"),
                "transportation": _numbered(day.get("transportation", [])),
                "accommodation": _numbered(day.get("accommodation", [])),
                "breakfast": _numbered(day.get("breakfast", [])),
                "lunch": _numbered(day.get("lunch", [])),
                "dinner": _numbered(day.get("dinner", [])),
                "attractions": _numbered(day.get("attractions", [])),
                "events": _numbered(day.get("events", [])),
            }
        )
    return json.dumps(formatted, ensure_ascii=True, indent=2)


def _pick_from_list(items, idx):
    if not items:
        return "-"
    if idx is None:
        return "-"
    try:
        idx_i = int(idx)
    except Exception:
        return "-"
    if idx_i < 0 or idx_i >= len(items):
        return "-"
    return items[idx_i]


def _resolve_day_plan_from_idx(day_candidates: dict, day_choice: dict) -> dict:
    transport = _pick_from_list(day_candidates["transportation"], day_choice.get("transportation_idx"))
    accommodation = _pick_from_list(day_candidates["accommodation"], day_choice.get("accommodation_idx"))
    breakfast = _pick_from_list(day_candidates["breakfast"], day_choice.get("breakfast_idx"))
    lunch = _pick_from_list(day_candidates["lunch"], day_choice.get("lunch_idx"))
    dinner = _pick_from_list(day_candidates["dinner"], day_choice.get("dinner_idx"))
    event = _pick_from_list(day_candidates["events"], day_choice.get("event_idx"))

    attr_idxs = day_choice.get("attraction_idxs") or []
    if not isinstance(attr_idxs, list):
        attr_idxs = [attr_idxs]
    attractions = []
    for idx in attr_idxs:
        name = _pick_from_list(day_candidates["attractions"], idx)
        if name and name != "-" and name not in attractions:
            attractions.append(name)
        if len(attractions) >= 2:
            break

    return {
        "transportation": transport,
        "accommodation": accommodation,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner,
        "attractions": attractions,
        "event": event,
        "from": day_candidates.get("from"),
        "to": day_candidates.get("to"),
    }


def _verify_choice(choice: dict, candidates_payload: dict) -> list:
    issues = []
    days_list = candidates_payload.get("days", [])
    if not isinstance(choice, dict):
        return ["choice_not_dict"]
    choice_days = choice.get("days")
    if not isinstance(choice_days, list):
        return ["choice_days_missing"]
    by_day = {}
    for d in choice_days:
        if isinstance(d, dict) and "day" in d:
            try:
                by_day[int(d.get("day"))] = d
            except Exception:
                continue
    for day_info in days_list:
        day = int(day_info.get("day"))
        if day not in by_day:
            issues.append(f"missing_day_{day}")
            continue
        dchoice = by_day[day]
        for slot_key, cand_key in (
            ("transportation_idx", "transportation"),
            ("accommodation_idx", "accommodation"),
            ("breakfast_idx", "breakfast"),
            ("lunch_idx", "lunch"),
            ("dinner_idx", "dinner"),
            ("event_idx", "events"),
        ):
            idx = dchoice.get(slot_key)
            if idx is None:
                issues.append(f"day{day}_{slot_key}_missing")
                continue
            try:
                idx_i = int(idx)
            except Exception:
                issues.append(f"day{day}_{slot_key}_not_int")
                continue
            cand_len = len(day_info.get(cand_key, []))
            if idx_i < 0 or idx_i >= cand_len:
                issues.append(f"day{day}_{slot_key}_out_of_range")
        attr_idxs = dchoice.get("attraction_idxs", [])
        if not isinstance(attr_idxs, list):
            issues.append(f"day{day}_attraction_idxs_not_list")
        else:
            if len(attr_idxs) > 2:
                issues.append(f"day{day}_attraction_idxs_too_many")
            for aidx in attr_idxs:
                try:
                    aidx_i = int(aidx)
                except Exception:
                    issues.append(f"day{day}_attraction_idx_not_int")
                    continue
                cand_len = len(day_info.get("attractions", []))
                if aidx_i < 0 or aidx_i >= cand_len:
                    issues.append(f"day{day}_attraction_idx_out_of_range")
    return issues


def _apply_day_plan(env: TripCraftEnv, state, day_plan: dict) -> None:
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


def load_csv_data(filename):
    data = pd.read_csv(filename)
    return data

def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=str, default="3day")
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt4o")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--strategy", type=str, default="direct_og")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the reference_info.csv file")
    args = parser.parse_args()

    # Load data from CSV
    data = load_csv_data(args.csv_file)

    # Ensure 'query', 'reference_information' columns exist
    
    # Prepare the dataset
    query_data_list = data.to_dict(orient='records')

    # Define planner based on strategy
    if args.strategy == 'direct_og':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_direct_og)
    elif args.strategy == 'direct_param':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_direct_param)
    elif args.strategy == 'llm_direct':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_llm_direct)
    elif args.strategy == 'llm_cot':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_llm_cot_plan)
    elif args.strategy == 'llm_reflexion':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_llm_direct)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Iterate over data and generate results
    with get_openai_callback() as cb:
        for number, query_data in enumerate(tqdm(query_data_list, desc="Processing data")):
            if args.day == '3day':
                reference_information = query_data['reference_information']
            elif args.day == '5day':
                reference_information_1 = json.loads(query_data['reference_information_1'])
                reference_information_2 = json.loads(query_data['reference_information_2'])
                reference_information = json.dumps(reference_information_1 + reference_information_2)
            else:
                reference_information_1 = json.loads(query_data['reference_information_1'])
                reference_information_2 = json.loads(query_data['reference_information_2'])
                reference_information_3 = json.loads(query_data['reference_information_3'])
                reference_information = json.dumps(reference_information_1 + reference_information_2 + reference_information_3)
            if args.strategy in ['llm_direct', 'llm_cot', 'llm_reflexion']:
                row = row_from_dict(query_data, idx_default=number + 1)
                template = make_output_record_template(row)
                kb = build_unified_kb(row.org, row.ref_blocks)
                env = TripCraftEnv(row=row, kb=kb, topk=5)
                state = env.initial_state()
                template_json = json.dumps(template, ensure_ascii=False, indent=2)
                candidates_payload = _build_candidates_payload(row, kb, env, topk=5)
                candidates_json = _format_candidates_for_llm(candidates_payload)
                if args.strategy == 'llm_cot':
                    draft_plan = planner.run(
                        reference_information,
                        query_data['query'],
                        query_data['persona'],
                        template=template_json,
                        candidates=candidates_json,
                    )
                    fill_planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_llm_cot_fill)
                    planner_results = fill_planner.run(
                        draft_plan,
                        "",
                        "",
                        candidates=candidates_json,
                    )
                    choice = _safe_json_loads(planner_results)
                    if not choice:
                        repair_planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_llm_direct_repair)
                        repair_raw = repair_planner.run(
                            planner_results,
                            "",
                            "",
                            candidates=candidates_json,
                        )
                        choice = _safe_json_loads(repair_raw)
                    choice = choice or {"days": []}
                elif args.strategy == 'llm_reflexion':
                    planner_results = planner.run(
                        reference_information,
                        query_data['query'],
                        query_data['persona'],
                        template=template_json,
                        candidates=candidates_json,
                    )
                    choice = _safe_json_loads(planner_results)
                    issues = _verify_choice(choice, candidates_payload) if choice else ["invalid_json"]
                    if issues:
                        repair_planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_llm_reflexion_repair)
                        issues_text = "\n".join(issues)
                        repair_raw = repair_planner.run(
                            planner_results,
                            "",
                            "",
                            candidates=candidates_json,
                            issues=issues_text,
                        )
                        choice = _safe_json_loads(repair_raw)
                    choice = choice or {"days": []}
                else:
                    planner_results = planner.run(
                        reference_information,
                        query_data['query'],
                        query_data['persona'],
                        template=template_json,
                        candidates=candidates_json,
                    )
                    choice = _safe_json_loads(planner_results)
                    if not choice:
                        repair_planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt_llm_direct_repair)
                        repair_raw = repair_planner.run(
                            planner_results,
                            "",
                            "",
                            candidates=candidates_json,
                        )
                        choice = _safe_json_loads(repair_raw)
                    choice = choice or {"days": []}
                day_choices = {int(d.get("day")): d for d in choice.get("days", []) if isinstance(d, dict)}
                day_plans = {}
                for day_info in candidates_payload["days"]:
                    day = int(day_info["day"])
                    day_choice = day_choices.get(day, {})
                    day_plans[day] = _resolve_day_plan_from_idx(day_info, day_choice)
                    _apply_day_plan(env, state, day_plans[day])
                planner_results = fill_template_with_state(template, row, kb, state)
                for day in range(1, row.days + 1):
                    event = day_plans.get(day, {}).get("event", "-")
                    city = _stage_city_for_day(kb, day)
                    planner_results["plan"][day - 1]["event"] = _with_city(event, city) if event and event != "-" else "-"
            else:
                while True:
                    if args.strategy in ['react', 'reflexion']:
                        planner_results, scratchpad = planner.run(reference_information, query_data['query'], query['persona'])
                    else:
                        planner_results = planner.run(reference_information, query_data['query'],query_data['persona'])
                    if planner_results is not None:
                        break
            print(planner_results)

            # Ensure the directory exists
            output_dir = os.path.join(args.output_dir, args.set_type)
            os.makedirs(output_dir, exist_ok=True)

            # Load previous results if available
            result_file = os.path.join(output_dir, f'llama_generated_plan_{number+1}.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    result = json.load(f)
            else:
                result = [{}]

            # Store the new results
            # if args.strategy in ['react', 'reflexion']:
            #     result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results_logs'] = scratchpad
            
            result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results'] = planner_results

            # Write to JSON file
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=4)

        print(cb)
