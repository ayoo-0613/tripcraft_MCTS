import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import (
    planner_agent_prompt_direct_og,
    planner_agent_prompt_direct_param,
    react_planner_agent_prompt,
    plan_skeleton_prompt,
    plan_execute_prompt,
    verifier_repair_prompt,
    reflexion_repair_prompt,
    template_guidance_prompt_param,
    action_select_prompt,
    action_select_prompt_react,
    action_select_prompt_reflexion,
    action_select_prompt_batch,
    fixed_skeleton_direct_prompt,
    fixed_skeleton_cot_plan_prompt,
    fixed_skeleton_cot_execute_prompt,
    fixed_skeleton_react_prompt,
    fixed_skeleton_reflexion_prompt,
)
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
# from langchain_community.llms import OpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from env import ReactEnv,ReactReflectEnv
import tiktoken
import re
import json
import openai
import time
from enum import Enum
from typing import Any, Dict, List, Union, Literal, Optional, Tuple
# from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

_EVAL_IMPORT_CWD = os.getcwd()
try:
    from evaluation.commonsense_constraint import evaluation as commonsense_eval
    from evaluation.hard_constraint import evaluation as hard_eval
finally:
    os.chdir(_EVAL_IMPORT_CWD)


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# openai.api_key = OPENAI_API_KEY
# GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

def _resolve_ollama_model(model_name: str) -> Optional[str]:
    if model_name.startswith("ollama:"):
        return model_name.split(":", 1)[1]
    if model_name == "ollama":
        return os.environ.get("OLLAMA_MODEL")
    return None


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout_sec: float = 999.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def chat(self, prompt: str) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content")
        return str(content or "")


class LLMRunner:
    def __init__(self, model_name: str = "gpt-3.5-turbo-1106") -> None:
        self.model_name = model_name
        self.ollama_model = _resolve_ollama_model(model_name)
        self.ollama_client = None
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if model_name.startswith("ollama") and not self.ollama_model:
            raise ValueError("Ollama model name missing. Set OLLAMA_MODEL or use model_name=ollama:<model>.")

        if model_name in ["qwen", "phi4"]:
            model_path = {
                "qwen": "Qwen/Qwen2.5-7B-Instruct",
                "phi4": "microsoft/Phi-4-mini-instruct",
            }[model_name]
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload",
                attn_implementation="flash_attention_2",
            )
            self.llm = None
        elif self.ollama_model:
            self.ollama_client = OllamaChatClient(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=self.ollama_model,
                timeout_sec=float(os.environ.get("OLLAMA_TIMEOUT", "999")),
            )
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=4096,
                openai_api_key=OPENAI_API_KEY,
            )

    def chat(self, prompt: str) -> str:
        if len(self.enc.encode(prompt)) > 12000:
            return "Max Token Length Exceeded."
        if self.model_name in ["qwen", "phi4"]:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=3072)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response_start = generated_text.find(prompt)
            if response_start != -1:
                generated_text = generated_text[response_start + len(prompt):].strip()
            return generated_text
        if self.ollama_client:
            return self.ollama_client.chat(prompt)
        if self.model_name == "gpt-4o":
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
                api_key=OPENAI_API_KEY,
            )
            return response["choices"][0]["message"]["content"]
        return self.llm([HumanMessage(content=prompt)]).content

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


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 agent_prompt: PromptTemplate = planner_agent_prompt_direct_og,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.ollama_model = _resolve_ollama_model(model_name)
        self.ollama_client = None
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if model_name.startswith("ollama") and not self.ollama_model:
            raise ValueError("Ollama model name missing. Set OLLAMA_MODEL or use model_name=ollama:<model>.")

        if model_name in ['qwen','phi4']:
            model_path = {
                'qwen': "Qwen/Qwen2.5-7B-Instruct",
                'phi4': "microsoft/Phi-4-mini-instruct"
            }[model_name]
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload",  # Enables CPU offloading
                attn_implementation="flash_attention_2"  # Speeds up inference
            )
        elif self.ollama_model:
            self.ollama_client = OllamaChatClient(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=self.ollama_model,
                timeout_sec=float(os.environ.get("OLLAMA_TIMEOUT", "999")),
            )
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=4096, openai_api_key=OPENAI_API_KEY)
        
        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, persona, log_file=None) -> str:
        if log_file:
            log_file.write('\n---------------Planner\n' + self._build_agent_prompt(text, query, persona))
        
        prompt = self._build_agent_prompt(text, query, persona)
        
        if self.model_name in ['qwen','phi4']:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            # print(self.model.generation_config)
            output = self.model.generate(**inputs, max_new_tokens=3072) #do_sample=False) # temperature=0.0) #equivalent
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            response_start = generated_text.find(prompt)
            if response_start != -1:
                generated_text = generated_text[response_start + len(prompt):].strip()
            
            return generated_text
        elif self.ollama_client:
            return self.ollama_client.chat(prompt)
        else:
            if len(self.enc.encode(prompt)) > 12000:
                return 'Max Token Length Exceeded.'
            elif self.model_name == 'gpt-4o':
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4096,
                    api_key=OPENAI_API_KEY
                )
                return response['choices'][0]['message']['content']
            else:
                return self.llm([HumanMessage(content=prompt)]).content

    def _build_agent_prompt(self, text, query, persona) -> str:
        return self.agent_prompt.format(text=text, query=query, persona=persona)


REQUIRED_PLAN_KEYS = [
    "days",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
    "event",
    "point_of_interest_list",
]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\\n", "", text)
        text = re.sub(r"\\n```$", "", text)
    return text.strip()


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    if text is None:
        return None
    text = _strip_code_fence(str(text).strip())
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
            obj = eval(chunk, {"__builtins__": {}})
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return None


def _normalize_plan(plan: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(plan, list):
        return None
    normalized = []
    for idx, item in enumerate(plan, start=1):
        if not isinstance(item, dict):
            return None
        norm = {}
        for key in REQUIRED_PLAN_KEYS:
            value = item.get(key, "-")
            if key == "days":
                try:
                    value = int(value)
                except Exception:
                    value = idx
            elif isinstance(value, list):
                value = "; ".join([str(v) for v in value])
            elif value is None:
                value = "-"
            else:
                value = str(value)
            norm[key] = value
        normalized.append(norm)
    return normalized


def _coerce_query_data(query_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not query_data:
        return None
    query_data = dict(query_data)
    local_constraint = query_data.get("local_constraint")
    if isinstance(local_constraint, str):
        try:
            query_data["local_constraint"] = eval(local_constraint, {"__builtins__": {}})
        except Exception:
            pass
    return query_data


def _collect_failures(query_data: Optional[Dict[str, Any]], plan: List[Dict[str, Any]]) -> List[str]:
    if not query_data:
        return ["Missing query data for constraint evaluation."]
    failures = []
    commonsense_info = commonsense_eval(query_data, plan)
    hard_info = hard_eval(query_data, plan)
    for info in (commonsense_info, hard_info):
        if not info:
            continue
        for key, value in info.items():
            if value is None:
                continue
            ok = value[0]
            msg = value[1] if len(value) > 1 else ""
            if ok is False:
                failures.append(f"{key}: {msg}")
    return failures


def _parse_choice_index(text: str, max_index: int) -> Optional[int]:
    if text is None:
        return None
    cleaned = str(text).strip()
    if cleaned == "":
        return None
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "choice" in obj:
            idx = int(obj["choice"])
            if 0 <= idx < max_index:
                return idx
    except Exception:
        pass
    match = re.search(r"-?\\d+", cleaned)
    if match:
        try:
            idx = int(match.group(0))
            if 0 <= idx < max_index:
                return idx
        except Exception:
            return None
    return None


def _action_view(action: Dict[str, Any]) -> Dict[str, Any]:
    view: Dict[str, Any] = {"type": action.get("type")}
    if "name" in action:
        view["name"] = action.get("name")
    if "raw" in action:
        view["raw"] = action.get("raw")
    meta = action.get("meta") or {}
    if isinstance(meta, dict):
        for key in ("cuisines", "subcategories", "room_type", "house_rules", "pricing_value", "avg_cost", "mode"):
            if key in meta:
                view[key] = meta.get(key)
    return view


def _parse_choice_list(text: str, max_len: int) -> List[Optional[int]]:
    parsed = _extract_json_array(text)
    if not isinstance(parsed, list):
        return []
    choices: List[Optional[int]] = []
    for item in parsed[:max_len]:
        idx: Optional[int] = None
        if isinstance(item, dict) and "choice" in item:
            try:
                idx = int(item["choice"])
            except Exception:
                idx = None
        elif isinstance(item, int):
            idx = item
        elif isinstance(item, str):
            match = re.search(r"-?\\d+", item)
            if match:
                try:
                    idx = int(match.group(0))
                except Exception:
                    idx = None
        choices.append(idx)
    return choices


def _repair_choice_list(text: str, expected_len: int) -> List[Optional[int]]:
    choices = _parse_choice_list(text, expected_len)
    if len(choices) >= expected_len:
        return choices[:expected_len]
    raw = str(text or "")
    nums = [int(m.group(0)) for m in re.finditer(r"-?\d+", raw)]
    if nums:
        return nums[:expected_len]
    return choices


def _repair_choice_index(text: str, max_index: int) -> Optional[int]:
    idx = _parse_choice_index(text, max_index)
    if idx is not None:
        return idx
    raw = str(text or "")
    match = re.search(r"-?\d+", raw)
    if not match:
        return None
    try:
        idx = int(match.group(0))
    except Exception:
        return None
    if idx < 0 or idx >= max_index:
        return None
    return idx


def _pad_choices(choices: List[Optional[int]], length: int) -> List[Optional[int]]:
    padded = list(choices[:length])
    while len(padded) < length:
        padded.append(0)
    return padded


def _action_key(action: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(action.get("type") or ""),
        str(action.get("name") or action.get("raw") or action.get("eval_poi_name") or "-"),
    )


def _match_action_index(
    choice_idx: Optional[int],
    reference_actions: List[Dict[str, Any]],
    current_actions: List[Dict[str, Any]],
) -> int:
    if choice_idx is None or choice_idx < 0 or choice_idx >= len(reference_actions):
        return 0
    target = reference_actions[choice_idx]
    target_key = _action_key(target)
    for idx, action in enumerate(current_actions):
        if _action_key(action) == target_key:
            return idx
    return 0


def _state_fingerprint(state: Any) -> Tuple[Any, ...]:
    return (
        getattr(state, "day", None),
        getattr(state, "substep", None),
        getattr(state, "time_cursor_min", None),
        getattr(state, "budget_used", None),
        getattr(state, "done", None),
    )


def _apply_action_safe(env: Any, state: Any, action: Dict[str, Any]) -> Any:
    before = _state_fingerprint(state)
    state = env.apply_action(state, action, record_trace=False)
    if _state_fingerprint(state) == before:
        state = env.apply_action(state, {"type": "end_day"}, record_trace=False)
    return state


def _candidate_actions_from_kb(
    env: Any,
    row: Any,
    kb: Any,
    state: Any,
    topk: int,
) -> List[Dict[str, Any]]:
    from mcts_baseline.retrieval import topk_accommodations, topk_attractions, topk_restaurants

    slot = env._slot_name(state)
    stage = env._stage_for_day(state.day)

    used_restaurants = set()
    used_attractions = set()
    for i, d in enumerate(state.drafts):
        stage_i = env._stage_for_day(i + 1)
        city = stage_i.city if stage_i is not None else ""
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
        current_city, frm, to = env._city_movement_for_day(state.day)
        date = row.date[state.day - 1] if state.day - 1 < len(row.date) else ""
        options = [t for t in kb.transports if t.frm == frm and t.to == to and (t.date is None or t.date == date)]
        if not options:
            options = [t for t in kb.transports if t.frm == frm and t.to == to]
        transport_constraint = (row.local_constraint or {}).get("transportation")
        if transport_constraint == "no flight":
            options = [t for t in options if t.mode != "flight"]
        elif transport_constraint == "no self-driving":
            options = [t for t in options if t.mode != "self-driving"]

        def rank(mode: str) -> int:
            return {"flight": 3, "self-driving": 2, "taxi": 1}.get(mode, 0)

        options = sorted(options, key=lambda t: rank(t.mode), reverse=True)[:topk]
        if not options:
            return [{"type": "set_transport", "raw": "-", "from": frm, "to": to, "candidates": ["-"]}]
        cand_raws = [t.raw for t in options]
        return [
            {
                "type": "set_transport",
                "raw": t.raw,
                "from": frm,
                "to": to,
                "eval_poi_name": "-",
                "meta": {"cost": t.cost, "mode": t.mode, "duration_min": t.duration_min},
                "candidates": cand_raws,
            }
            for t in options
        ]

    if slot == "accommodation":
        if stage is None:
            return [{"type": "set_accommodation", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
        cands = topk_accommodations(stage, local_constraint=row.local_constraint or {}, k=topk)
        if not cands:
            return [{"type": "set_accommodation", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
        names = [c["name"] for c in cands]
        return [
            {
                "type": "set_accommodation",
                "name": c["name"],
                "eval_poi_name": c["name"],
                "meta": c,
                "candidates": names,
            }
            for c in cands
        ]

    if slot in {"breakfast", "lunch", "dinner"}:
        if stage is None:
            return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
        cands = topk_restaurants(stage, meal=slot, local_constraint=row.local_constraint or {}, k=topk)
        cands = [c for c in cands if (c["name"], stage.city) not in used_restaurants]
        if not cands:
            return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
        names = [c["name"] for c in cands]
        actions = [
            {
                "type": f"set_{slot}",
                "name": c["name"],
                "eval_poi_name": c["name"],
                "meta": c,
                "candidates": names,
            }
            for c in cands
        ]
        return actions

    if slot in {"attraction1", "attraction2"}:
        if stage is None:
            return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
        cands = topk_attractions(stage, local_constraint=row.local_constraint or {}, k=topk)
        cands = [c for c in cands if (c["name"], stage.city) not in used_attractions]
        if not cands:
            return [{"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]}]
        names = [c["name"] for c in cands]
        actions = [
            {
                "type": f"add_{slot}",
                "name": c["name"],
                "eval_poi_name": c["name"],
                "meta": c,
                "candidates": names,
            }
            for c in cands
        ]
        if slot == "attraction2":
            actions.append({"type": f"skip_{slot}", "name": "-", "eval_poi_name": "-", "candidates": ["-"]})
        return actions

    if slot == "end_day":
        return [{"type": "end_day", "eval_poi_name": "-"}]

    return []


def _build_fixed_skeleton_steps(
    env: Any,
    row: Any,
    kb: Any,
    template_list: List[Dict[str, Any]],
    topk: int,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    preview_state = env.initial_state()
    steps: List[Dict[str, Any]] = []
    actions_by_step: List[List[Dict[str, Any]]] = []
    while not env.is_terminal(preview_state):
        slot = env._slot_name(preview_state)
        actions = _candidate_actions_from_kb(env, row, kb, preview_state, topk)
        if not actions:
            preview_state = _apply_action_safe(env, preview_state, {"type": "end_day"})
            continue
        if slot == "end_day":
            preview_state = _apply_action_safe(env, preview_state, actions[0])
            continue
        day_idx = max(preview_state.day - 1, 0)
        day_template = template_list[day_idx] if day_idx < len(template_list) else {}
        candidates = [dict(index=i, **_action_view(a)) for i, a in enumerate(actions)]
        steps.append(
            {
                "day": preview_state.day,
                "slot": slot,
                "template_day": day_template or {},
                "candidates": candidates,
            }
        )
        actions_by_step.append(actions)
        preview_state = _apply_action_safe(env, preview_state, actions[0])
    return steps, actions_by_step


def _apply_choice_sequence(
    env: Any,
    row: Any,
    kb: Any,
    actions_by_step: List[List[Dict[str, Any]]],
    choices: List[Optional[int]],
    topk: int,
) -> Any:
    state = env.initial_state()
    step_idx = 0
    while not env.is_terminal(state):
        slot = env._slot_name(state)
        actions = _candidate_actions_from_kb(env, row, kb, state, topk)
        if not actions:
            state = _apply_action_safe(env, state, {"type": "end_day"})
            continue
        if slot == "end_day":
            state = _apply_action_safe(env, state, actions[0])
            continue
        if step_idx < len(actions_by_step):
            ref_actions = actions_by_step[step_idx]
            choice_idx = choices[step_idx] if step_idx < len(choices) else None
            idx = _match_action_index(choice_idx, ref_actions, actions)
        else:
            idx = 0
        state = _apply_action_safe(env, state, actions[idx])
        step_idx += 1
    return state


def _build_partial_plan_snapshot(
    template_list: List[Dict[str, Any]],
    state: Any,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for i, draft in enumerate(state.drafts):
        template_day = template_list[i] if i < len(template_list) else {}
        current_city = draft.current_city if draft.current_city != "-" else template_day.get("current_city", "-")
        attractions = draft.attractions or []
        plan.append(
            {
                "days": i + 1,
                "current_city": current_city,
                "transportation": draft.transportation,
                "breakfast": draft.breakfast,
                "attraction": "; ".join(attractions) if attractions else "-",
                "lunch": draft.lunch,
                "dinner": draft.dinner,
                "accommodation": draft.accommodation,
                "event": draft.event,
                "point_of_interest_list": "-",
            }
        )
    return plan


def _verify_plan(query_data: Optional[Dict[str, Any]], plan: Any) -> List[str]:
    if not isinstance(plan, list) or not plan:
        return ["Plan is not valid JSON array with required keys."]
    failures: List[str] = []
    for idx, day in enumerate(plan, start=1):
        if not isinstance(day, dict):
            failures.append(f"Day {idx} is not a JSON object.")
            continue
        missing = [k for k in REQUIRED_PLAN_KEYS if k not in day]
        if missing:
            failures.append(f"Day {idx} missing keys: {', '.join(missing)}")
    failures += _collect_failures(query_data, plan)
    return failures


class TemplateActionPlanner:
    def __init__(
        self,
        model_name: str,
        template_prompt: PromptTemplate,
        action_prompt: PromptTemplate,
        action_prompt_react: PromptTemplate,
        action_prompt_reflexion: PromptTemplate,
        action_prompt_one_shot: PromptTemplate,
        action_strategy: str = "direct",
        topk: int = 5,
        one_shot: bool = False,
    ) -> None:
        self.runner = LLMRunner(model_name=model_name)
        self.template_prompt = template_prompt
        self.action_prompt = action_prompt
        self.action_prompt_react = action_prompt_react
        self.action_prompt_reflexion = action_prompt_reflexion
        self.action_prompt_one_shot = action_prompt_one_shot
        self.action_strategy = str(action_strategy or "direct").lower()
        self.topk = topk
        self.one_shot = one_shot

    def run(self, text, query, persona, query_data=None) -> str:
        query_data = _coerce_query_data(query_data)
        from mcts_baseline.io import row_from_dict
        from mcts_baseline.templater import make_output_record_template

        row = row_from_dict(query_data or {}, idx_default=1)
        fallback_record = make_output_record_template(row)

        if not query_data:
            return fallback_record
        if not row.ref_blocks:
            return fallback_record

        try:
            from mcts_baseline.env import TripCraftEnv
            from mcts_baseline.formatter import fill_template_with_state
            from mcts_baseline.ref_parser import build_unified_kb

            kb = build_unified_kb(row.org, row.ref_blocks)
            env = TripCraftEnv(row=row, kb=kb, topk=self.topk)
            state = env.initial_state()
            template_list = make_output_record_template(row)["plan"]
            for d in range(1, row.days + 1):
                cc, _, _ = env._city_movement_for_day(d)
                template_list[d - 1]["current_city"] = cc

            if self.one_shot:
                preview_state = env.initial_state()
                steps: List[Dict[str, Any]] = []
                actions_by_step: List[List[Dict[str, Any]]] = []
                while not env.is_terminal(preview_state):
                    slot = env._slot_name(preview_state)
                    actions = env.legal_actions(preview_state, topk=self.topk)
                    if not actions:
                        preview_state = env.apply_action(preview_state, {"type": "end_day"}, record_trace=False)
                        continue
                    if slot == "end_day":
                        preview_state = env.apply_action(preview_state, actions[0], record_trace=False)
                        continue
                    day_idx = max(preview_state.day - 1, 0)
                    day_template = template_list[day_idx] if day_idx < len(template_list) else {}
                    candidates = [dict(index=i, **_action_view(a)) for i, a in enumerate(actions)]
                    steps.append(
                        {
                            "day": preview_state.day,
                            "slot": slot,
                            "template_day": day_template or {},
                            "candidates": candidates,
                        }
                    )
                    actions_by_step.append(actions)
                    # Advance with a deterministic action to enumerate subsequent slots.
                    preview_state = env.apply_action(preview_state, actions[0], record_trace=False)

                prompt = self.action_prompt_one_shot.format(
                    text=text,
                    query=query,
                    persona=persona,
                    template=json.dumps(template_list, ensure_ascii=True),
                    steps=json.dumps(steps, ensure_ascii=True),
                )
                response = self.runner.chat(prompt)
                choice_list = _parse_choice_list(response or "", len(actions_by_step))

                state = env.initial_state()
                for i, actions in enumerate(actions_by_step):
                    idx = choice_list[i] if i < len(choice_list) else None
                    if idx is None or idx < 0 or idx >= len(actions):
                        idx = 0
                    chosen = actions[idx]
                    state = env.apply_action(state, chosen, record_trace=False)
            else:
                while not env.is_terminal(state):
                    slot = env._slot_name(state)
                    actions = env.legal_actions(state, topk=self.topk)
                    if not actions:
                        state = _apply_action_safe(env, state, {"type": "end_day"})
                        continue
                    if slot == "end_day":
                        state = _apply_action_safe(env, state, actions[0])
                        continue

                    day_idx = max(state.day - 1, 0)
                    day_template = template_list[day_idx] if day_idx < len(template_list) else {}
                    candidates = [dict(index=i, **_action_view(a)) for i, a in enumerate(actions)]
                    base_kwargs = dict(
                        day=state.day,
                        slot=slot,
                        text=text,
                        persona=persona,
                        query=query,
                        template_day=json.dumps(day_template or {}, ensure_ascii=True),
                        candidates=json.dumps(candidates, ensure_ascii=True),
                    )
                    if self.action_strategy == "react":
                        prompt = self.action_prompt_react.format(**base_kwargs)
                        response = self.runner.chat(prompt)
                        idx = _parse_choice_index(response, len(actions))
                    elif self.action_strategy == "reflexion":
                        prompt = self.action_prompt.format(**base_kwargs)
                        response = self.runner.chat(prompt)
                        idx = _parse_choice_index(response, len(actions))
                        initial_choice = idx if idx is not None else 0
                        prompt = self.action_prompt_reflexion.format(initial_choice=initial_choice, **base_kwargs)
                        response = self.runner.chat(prompt)
                        idx = _parse_choice_index(response, len(actions)) if response else idx
                    else:
                        prompt = self.action_prompt.format(**base_kwargs)
                        response = self.runner.chat(prompt)
                        idx = _parse_choice_index(response, len(actions))
                    chosen = actions[idx] if idx is not None else actions[0]
                    state = env.apply_action(state, chosen, record_trace=False)

            template = make_output_record_template(row)
            record = fill_template_with_state(template, row, kb, state)
            return record
        except Exception:
            return fallback_record


def _build_constraints_summary(row: Any) -> str:
    parts = [
        f"org={getattr(row, 'org', '')}",
        f"dest={getattr(row, 'dest', '')}",
        f"days={getattr(row, 'days', '')}",
        f"people={getattr(row, 'people_number', '')}",
        f"budget={getattr(row, 'budget', '')}",
    ]
    dates = getattr(row, "date", None)
    if dates:
        parts.append(f"dates={dates}")
    local_constraint = getattr(row, "local_constraint", None)
    if local_constraint:
        try:
            parts.append(f"local_constraint={json.dumps(local_constraint, ensure_ascii=True)}")
        except Exception:
            parts.append(f"local_constraint={local_constraint}")
    return "; ".join([p for p in parts if p and p != "None"])


def _parse_value_object(text: str) -> Optional[str]:
    cleaned = _strip_code_fence(str(text or "").strip())
    if cleaned == "":
        return None
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "value" in obj:
            return str(obj["value"])
        if isinstance(obj, str):
            return obj
    except Exception:
        pass
    return cleaned


class FixedSkeletonPlanner:
    def __init__(
        self,
        model_name: str,
        strategy: str,
        direct_prompt: PromptTemplate = fixed_skeleton_direct_prompt,
        cot_plan_prompt: PromptTemplate = fixed_skeleton_cot_plan_prompt,
        cot_execute_prompt: PromptTemplate = fixed_skeleton_cot_execute_prompt,
        react_prompt: PromptTemplate = fixed_skeleton_react_prompt,
        reflexion_prompt: PromptTemplate = fixed_skeleton_reflexion_prompt,
        topk: int = 5,
    ) -> None:
        self.runner = LLMRunner(model_name=model_name)
        self.strategy = str(strategy or "direct").lower()
        self.direct_prompt = direct_prompt
        self.cot_plan_prompt = cot_plan_prompt
        self.cot_execute_prompt = cot_execute_prompt
        self.react_prompt = react_prompt
        self.reflexion_prompt = reflexion_prompt
        self.topk = topk

    def run(self, text, query, persona, query_data=None) -> str:
        query_data = _coerce_query_data(query_data)
        from mcts_baseline.io import row_from_dict
        from mcts_baseline.templater import make_output_record_template

        row = row_from_dict(query_data or {}, idx_default=1)
        fallback_record = make_output_record_template(row)

        try:
            template = make_output_record_template(row)
            template_list = template["plan"]
            template_json = json.dumps(template_list, ensure_ascii=True)

            if self.strategy == "direct":
                prompt = self.direct_prompt.format(
                    text=text,
                    query=query,
                    persona=persona,
                    plan_json=template_json,
                )
                response = self.runner.chat(prompt)
                parsed = _extract_json_array(response or "")
                plan_list = _normalize_plan(parsed) if parsed is not None else None
                if plan_list is None:
                    plan_list = template_list
            elif self.strategy == "cot":
                plan_prompt = self.cot_plan_prompt.format(
                    text=text,
                    query=query,
                    persona=persona,
                    plan_json=template_json,
                )
                slot_plan = self.runner.chat(plan_prompt)
                execute_prompt = self.cot_execute_prompt.format(
                    text=text,
                    query=query,
                    persona=persona,
                    plan_json=template_json,
                    slot_plan=slot_plan,
                )
                response = self.runner.chat(execute_prompt)
                parsed = _extract_json_array(response or "")
                plan_list = _normalize_plan(parsed) if parsed is not None else None
                if plan_list is None:
                    plan_list = template_list
            elif self.strategy == "react":
                plan_list = json.loads(template_json)
                slot_order = [
                    "current_city",
                    "transportation",
                    "breakfast",
                    "attraction",
                    "lunch",
                    "dinner",
                    "accommodation",
                    "event",
                    "point_of_interest_list",
                ]
                for day_idx, day in enumerate(plan_list, start=1):
                    for slot in slot_order:
                        partial_plan = json.dumps(plan_list, ensure_ascii=True)
                        prompt = self.react_prompt.format(
                            text=text,
                            query=query,
                            persona=persona,
                            partial_plan=partial_plan,
                            day=day_idx,
                            slot=slot,
                        )
                        response = self.runner.chat(prompt)
                        value = _parse_value_object(response)
                        day[slot] = value if value is not None else "-"
            elif self.strategy == "reflexion":
                prompt = self.direct_prompt.format(
                    text=text,
                    query=query,
                    persona=persona,
                    plan_json=template_json,
                )
                response = self.runner.chat(prompt)
                parsed = _extract_json_array(response or "")
                plan_list = _normalize_plan(parsed) if parsed is not None else None
                if plan_list is None:
                    plan_list = template_list
                failures = _verify_plan(query_data, plan_list) if query_data else []
                if failures:
                    repair_prompt = self.reflexion_prompt.format(
                        text=text,
                        query=query,
                        persona=persona,
                        failures="; ".join(failures),
                        plan_json=json.dumps(plan_list, ensure_ascii=True),
                        plan_json_template=template_json,
                    )
                    repair_response = self.runner.chat(repair_prompt)
                    parsed = _extract_json_array(repair_response or "")
                    repaired = _normalize_plan(parsed) if parsed is not None else None
                    if repaired is not None:
                        plan_list = repaired
            else:
                raise ValueError(f"Unknown fixed-skeleton strategy: {self.strategy}")

            record = template
            record["plan"] = plan_list
            return record
        except Exception:
            return fallback_record


class VerifierRepairPlanner:
    def __init__(
        self,
        model_name: str,
        agent_prompt: PromptTemplate,
        repair_prompt: PromptTemplate,
        max_rounds: int = 2,
    ) -> None:
        self.runner = LLMRunner(model_name=model_name)
        self.agent_prompt = agent_prompt
        self.repair_prompt = repair_prompt
        self.max_rounds = max_rounds

    def run(self, text, query, persona, query_data=None) -> str:
        prompt = self.agent_prompt.format(text=text, query=query, persona=persona)
        plan_text = self.runner.chat(prompt)
        query_data = _coerce_query_data(query_data)
        for _ in range(self.max_rounds):
            parsed = _extract_json_array(plan_text)
            normalized = _normalize_plan(parsed) if parsed is not None else None
            if normalized is None:
                failures = ["Plan is not valid JSON array with required keys."]
            else:
                failures = _collect_failures(query_data, normalized)
            if not failures:
                return plan_text
            plan_json = json.dumps(normalized or plan_text, ensure_ascii=True)
            repair = self.repair_prompt.format(
                text=text,
                query=query,
                persona=persona,
                plan_json=plan_json,
                failures="; ".join(failures),
            )
            plan_text = self.runner.chat(repair)
        return plan_text


class ReflexionPlanner(VerifierRepairPlanner):
    pass


class PlanExecutePlanner:
    def __init__(
        self,
        model_name: str,
        plan_prompt: PromptTemplate,
        execute_prompt: PromptTemplate,
    ) -> None:
        self.runner = LLMRunner(model_name=model_name)
        self.plan_prompt = plan_prompt
        self.execute_prompt = execute_prompt

    def run(self, text, query, persona, query_data=None) -> str:
        skeleton = self.runner.chat(self.plan_prompt.format(text=text, query=query, persona=persona))
        execute = self.execute_prompt.format(
            text=text,
            query=query,
            persona=persona,
            plan_json=skeleton,
        )
        return self.runner.chat(execute)


class ReactPlanner:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.model_name = model_name
        self.ollama_model = _resolve_ollama_model(model_name)
        self.ollama_client = None

        if model_name.startswith("ollama") and not self.ollama_model:
            raise ValueError("Ollama model name missing. Set OLLAMA_MODEL or use model_name=ollama:<model>.")

        if self.ollama_model:
            self.ollama_client = OllamaChatClient(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=self.ollama_model,
                timeout_sec=float(os.environ.get("OLLAMA_TIMEOUT", "999")),
            )
            self.react_llm = None
        else:
            self.react_llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=1024,
                openai_api_key=OPENAI_API_KEY,
                model_kwargs={"stop": ["Action", "Thought", "Observation"]},
            )

        self.env = ReactEnv()
        self.query = None
        self.persona = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, persona, reset = True) -> None:

        self.query = query
        self.text = text
        self.persona = persona

        if reset:
            self.reset()

        while not (self.is_halted() or self.is_finished()):
            self.step()

        if not self.answer:
            force_prompt = (
                self._build_agent_prompt()
                + "\n\nYou must now finish. Output only: Finish[<final Travel Plan>]"
            )
            try:
                forced = format_step(self._call_llm(force_prompt))
            except Exception:
                forced = ""
            action_type, action_arg = parse_action(forced)
            if action_type == "Finish":
                self.answer = action_arg
            elif forced:
                self.answer = forced

        return self.answer, self.scratchpad

    def _call_llm(self, prompt: str) -> str:
        if self.ollama_client:
            return self.ollama_client.chat(prompt)
        return self.react_llm([HumanMessage(content=prompt)]).content

    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)

        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg

        else:
            observation = f'Action {action_type} is not supported.'

        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(self._call_llm(self._build_agent_prompt()))
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            persona = self.persona,
                            scratchpad = self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False


# class ReactReflectPlanner:
#     """
#     A question answering Self-Reflecting React Agent.
#     """
#     def __init__(self,
#                  agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
#                 reflect_prompt: PromptTemplate = reflect_prompt,
#                  model_name: str = 'gpt-3.5-turbo-1106',
#                  ) -> None:
        
#         self.agent_prompt = agent_prompt
#         self.reflect_prompt = reflect_prompt
#         if model_name in ['gemini']:
#             self.react_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
#             self.reflect_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
#         else:
#             self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
#             self.reflect_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
#         self.model_name = model_name
#         self.env = ReactReflectEnv()
#         self.query = None
#         self.max_steps = 30
#         self.reset()
#         self.finished = False
#         self.answer = ''
#         self.reflections: List[str] = []
#         self.reflections_str: str = ''
#         self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

#     def run(self, text, query, reset = True) -> None:

#         self.query = query
#         self.text = text

#         if reset:
#             self.reset()
        

#         while not (self.is_halted() or self.is_finished()):
#             self.step()
#             if self.env.is_terminated and not self.finished:
#                 self.reflect(ReflexionStrategy.REFLEXION)

        
#         return self.answer, self.scratchpad

    
#     def step(self) -> None:
#         # Think
#         self.scratchpad += f'\nThought {self.curr_step}:'
#         self.scratchpad += ' ' + self.prompt_agent()
#         print(self.scratchpad.split('\n')[-1])

#         # Act
#         self.scratchpad += f'\nAction {self.curr_step}:'
#         action = self.prompt_agent()
#         self.scratchpad += ' ' + action
#         print(self.scratchpad.split('\n')[-1])

#         # Observe
#         self.scratchpad += f'\nObservation {self.curr_step}: '

#         action_type, action_arg = parse_action(action)

#         if action_type == 'CostEnquiry':
#             try:
#                 input_arg = eval(action_arg)
#                 if type(input_arg) != dict:
#                     raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
#                 observation = f'Cost: {self.env.run(input_arg)}'
#             except SyntaxError:
#                 observation = f'The sub plan can not be parsed into json format, please check.'
#             except ValueError as e:
#                 observation = str(e)
        
#         elif action_type == 'Finish':
#             self.finished = True
#             observation = f'The plan is finished.'
#             self.answer = action_arg
        
#         else:
#             observation = f'Action {action_type} is not supported.'
        
#         self.curr_step += 1

#         self.scratchpad += observation
#         print(self.scratchpad.split('\n')[-1])

#     def reflect(self, strategy: ReflexionStrategy) -> None:
#         print('Reflecting...')
#         if strategy == ReflexionStrategy.REFLEXION: 
#             self.reflections += [self.prompt_reflection()]
#             self.reflections_str = format_reflections(self.reflections)
#         else:
#             raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
#         print(self.reflections_str)

#     def prompt_agent(self) -> str:
#         while True:
#             try:
#                 if self.model_name in ['gemini']:
#                     return format_step(self.react_llm.invoke(self._build_agent_prompt()).content)
#                 else:
#                     return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
#             except:
#                 catch_openai_api_error()
#                 print(self._build_agent_prompt())
#                 print(len(self.enc.encode(self._build_agent_prompt())))
#                 time.sleep(5)
    
#     def prompt_reflection(self) -> str:
#         while True:
#             try:
#                 if self.model_name in ['gemini']:
#                     return format_step(self.reflect_llm.invoke(self._build_reflection_prompt()).content)
#                 else:
#                     return format_step(self.reflect_llm([HumanMessage(content=self._build_reflection_prompt())]).content)
#             except:
#                 catch_openai_api_error()
#                 print(self._build_reflection_prompt())
#                 print(len(self.enc.encode(self._build_reflection_prompt())))
#                 time.sleep(5)
    
#     def _build_agent_prompt(self) -> str:
#         return self.agent_prompt.format(
#                             query = self.query,
#                             text = self.text,
#                             scratchpad = self.scratchpad,
#                             reflections = self.reflections_str)
    
#     def _build_reflection_prompt(self) -> str:
#         return self.reflect_prompt.format(
#                             query = self.query,
#                             text = self.text,
#                             scratchpad = self.scratchpad)
    
#     def is_finished(self) -> bool:
#         return self.finished

#     def is_halted(self) -> bool:
#         return ((self.curr_step > self.max_steps) or (
#                     len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

#     def reset(self) -> None:
#         self.scratchpad = ''
#         self.answer = ''
#         self.curr_step = 1
#         self.finished = False
#         self.reflections = []
#         self.reflections_str = ''
#         self.env.reset()

class ReactReflectPlanner(ReactPlanner):
    pass


def format_step(step: str) -> str:
    return step.strip()

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string, flags=re.DOTALL)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

# def format_reflections(reflections: List[str],
#                         header: str = REFLECTION_HEADER) -> str:
#     if reflections == []:
#         return ''
#     else:
#         return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# if __name__ == '__main__':
    
