from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .env import State, TripCraftEnv
from .persona_utils import (
    average_persona_similarity,
    candidate_poi_name,
    get_persona_embeddings,
    get_poi_embedding,
    get_text_embedding,
    text_similarity,
)


@dataclass
class GuidanceConfig:
    mode: str = "none"  # none | heuristic | persona | llm | ollama
    endpoint: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    prior_prompt_path: Optional[str] = None
    value_prompt_path: Optional[str] = None
    timeout_sec: float = 10.0


_RESTAURANT_FEATURES: Optional[Tuple[Dict[Tuple[str, str], List[str]], Dict[str, List[str]]]] = None
_ATTRACTION_SUBCATEGORIES: Optional[Tuple[Dict[Tuple[str, str], List[str]], Dict[str, List[str]]]] = None
_LLM_PREF_WEIGHT = 0.1
_LLM_PREF_PROMPT_TEMPLATE = (
    "You are given a traveler persona and optional constraints.\n"
    "Generate compact preference keywords for restaurants and attractions.\n"
    "Return JSON only: {{\"food_preferences\":\"...\",\"attraction_preferences\":\"...\"}}\n"
    "Rules:\n"
    "- Use 5-12 short comma-separated keywords or phrases.\n"
    "- Focus on style, atmosphere, cuisine traits, and activity themes.\n"
    "- Do not include city names or POI names.\n"
    "- Keep each field under 200 characters.\n"
    "Persona: {persona}\n"
    "Local constraints: {local_constraint}\n"
    "Required cuisines: {required_cuisines}\n"
    "Required attraction types: {required_attraction_types}\n"
)


def _normalize_key(val: Any) -> str:
    return str(val or "").strip().lower()


def _normalize_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val if x is not None]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x is not None]
            except Exception:
                return [s]
        return [s]
    return []


def _merge_unique(primary: List[str], extra: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in primary + extra:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _list_to_text(items: List[str]) -> str:
    return ", ".join([str(x).strip() for x in items if str(x).strip()])


def _safe_json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        return str(value or "")


def _coerce_pref_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join([str(x).strip() for x in value if str(x).strip()])
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _load_restaurant_features() -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, List[str]]]:
    global _RESTAURANT_FEATURES
    if _RESTAURANT_FEATURES is not None:
        return _RESTAURANT_FEATURES
    try:
        import pandas as pd
        from utils.paths import tripcraft_db_root

        path = tripcraft_db_root() / "restaurants" / "cleaned_restaurant_details_2024.csv"
        df = pd.read_csv(path, usecols=["name", "City", "features"])
    except Exception:
        _RESTAURANT_FEATURES = ({}, {})
        return _RESTAURANT_FEATURES

    by_city: Dict[Tuple[str, str], List[str]] = {}
    by_name: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        name = _normalize_key(row.get("name"))
        if not name:
            continue
        city = _normalize_key(row.get("City"))
        features = _normalize_list(row.get("features"))
        if not features:
            continue
        by_name.setdefault(name, features)
        if city:
            by_city.setdefault((city, name), features)

    _RESTAURANT_FEATURES = (by_city, by_name)
    return _RESTAURANT_FEATURES


def _load_attraction_subcategories() -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, List[str]]]:
    global _ATTRACTION_SUBCATEGORIES
    if _ATTRACTION_SUBCATEGORIES is not None:
        return _ATTRACTION_SUBCATEGORIES
    try:
        import pandas as pd
        from utils.paths import tripcraft_db_root

        path = tripcraft_db_root() / "attraction" / "cleaned_attractions_final.csv"
        df = pd.read_csv(path, usecols=["name", "City", "subcategories"])
    except Exception:
        _ATTRACTION_SUBCATEGORIES = ({}, {})
        return _ATTRACTION_SUBCATEGORIES

    by_city: Dict[Tuple[str, str], List[str]] = {}
    by_name: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        name = _normalize_key(row.get("name"))
        if not name:
            continue
        city = _normalize_key(row.get("City"))
        subcats = _normalize_list(row.get("subcategories"))
        if not subcats:
            continue
        by_name.setdefault(name, subcats)
        if city:
            by_city.setdefault((city, name), subcats)

    _ATTRACTION_SUBCATEGORIES = (by_city, by_name)
    return _ATTRACTION_SUBCATEGORIES


def _lookup_restaurant_features(name: str, city: str) -> List[str]:
    by_city, by_name = _load_restaurant_features()
    name_key = _normalize_key(name)
    if not name_key:
        return []
    city_key = _normalize_key(city)
    if city_key:
        hit = by_city.get((city_key, name_key))
        if hit:
            return hit
    return by_name.get(name_key, [])


def _lookup_attraction_subcategories(name: str, city: str) -> List[str]:
    by_city, by_name = _load_attraction_subcategories()
    name_key = _normalize_key(name)
    if not name_key:
        return []
    city_key = _normalize_key(city)
    if city_key:
        hit = by_city.get((city_key, name_key))
        if hit:
            return hit
    return by_name.get(name_key, [])


def _action_city(action: Dict[str, Any], state: State, env: TripCraftEnv) -> str:
    meta = action.get("meta") or {}
    city = meta.get("city") or meta.get("City")
    if city:
        return str(city)
    if env and state and getattr(env, "kb", None) and env.kb.stages:
        idx = min((state.day - 1) // 2, len(env.kb.stages) - 1)
        return env.kb.stages[idx].city
    return ""


def _persona_text_similarity(text: str, persona_embeddings: Dict[str, Any]) -> Optional[float]:
    if not text:
        return None
    emb = get_text_embedding(text)
    if emb is None:
        return None
    return average_persona_similarity(emb, persona_embeddings)


def _combine_scores(name_score: Optional[float], extra_score: Optional[float]) -> float:
    if name_score is None and extra_score is None:
        return 0.0
    if name_score is None:
        return float(extra_score or 0.0)
    if extra_score is None:
        return float(name_score)
    return (float(name_score) + float(extra_score)) / 2.0


def _blend_scores(primary: Optional[float], secondary: Optional[float], weight: float) -> Optional[float]:
    if primary is None and secondary is None:
        return None
    if primary is None:
        return float(secondary or 0.0)
    if secondary is None:
        return float(primary)
    return (1.0 - weight) * float(primary) + weight * float(secondary)


def _normalize_llm_prefs(obj: Any) -> Dict[str, str]:
    if not isinstance(obj, dict):
        return {}
    food = _coerce_pref_text(
        obj.get("food_preferences")
        or obj.get("food")
        or obj.get("restaurant_preferences")
        or obj.get("dining_preferences")
    )
    attraction = _coerce_pref_text(
        obj.get("attraction_preferences")
        or obj.get("attractions")
        or obj.get("attraction_types")
        or obj.get("activity_preferences")
    )
    out: Dict[str, str] = {}
    if food:
        out["food_preferences"] = food
    if attraction:
        out["attraction_preferences"] = attraction
    return out


def _render_llm_pref_prompt(persona: str, env: TripCraftEnv) -> str:
    return _LLM_PREF_PROMPT_TEMPLATE.format(
        persona=persona,
        local_constraint=_safe_json_dump(getattr(env.row, "local_constraint", None)),
        required_cuisines=_safe_json_dump(getattr(env, "required_cuisines", [])),
        required_attraction_types=_safe_json_dump(getattr(env, "required_attraction_types", [])),
    )


class GuidanceModel:
    def prior(self, state: State, actions: List[Dict[str, Any]], env: TripCraftEnv) -> List[float]:
        raise NotImplementedError

    def value(self, state: State, env: TripCraftEnv) -> float:
        raise NotImplementedError


class NullGuidance(GuidanceModel):
    def prior(self, state: State, actions: List[Dict[str, Any]], env: TripCraftEnv) -> List[float]:
        if not actions:
            return []
        return [1.0 / len(actions)] * len(actions)

    def value(self, state: State, env: TripCraftEnv) -> float:
        return 0.0


class HeuristicGuidance(GuidanceModel):
    def _score_action(self, action: Dict[str, Any], env: TripCraftEnv) -> float:
        meta = action.get("meta") or {}
        score = 0.0
        if action["type"] == "set_transport":
            mode = meta.get("mode") or ""
            score += {"flight": 3.0, "taxi": 2.0, "self-driving": 1.0}.get(mode, 0.5)
        if action["type"] == "set_accommodation":
            rating = meta.get("rating") or 0.0
            pricing = meta.get("pricing_value") or 0.0
            score += float(rating)
            score -= float(pricing) / 200.0 if pricing else 0.0
        if action["type"].startswith("set_") and "avg_cost" in meta:
            rating = meta.get("rating") or 0.0
            score += float(rating)
        if action["type"].startswith("add_"):
            visit = meta.get("visit_duration") or 0.0
            score += float(visit)
        if action["type"].startswith("set_") and action["type"].endswith(("breakfast", "lunch", "dinner")):
            cuisines_req = env.required_cuisines
            if cuisines_req:
                cuisines = meta.get("cuisines") or []
                if any(c in cuisines for c in cuisines_req):
                    score += 1.0
        if action["type"].startswith("add_"):
            attr_req = env.required_attraction_types
            if attr_req:
                subcats = meta.get("subcategories") or []
                if any(a in subcats for a in attr_req):
                    score += 1.0
        return score

    def prior(self, state: State, actions: List[Dict[str, Any]], env: TripCraftEnv) -> List[float]:
        if not actions:
            return []
        scores = [self._score_action(a, env) for a in actions]
        max_s = max(scores)
        exp = [math.exp(s - max_s) for s in scores]
        total = sum(exp)
        if total <= 0:
            return [1.0 / len(actions)] * len(actions)
        return [v / total for v in exp]

    def value(self, state: State, env: TripCraftEnv) -> float:
        filled = 0
        total = env.row.days * 6
        for d in state.drafts:
            for key in ("breakfast", "lunch", "dinner", "accommodation"):
                if getattr(d, key) != "-":
                    filled += 1
            if d.attractions:
                filled += 1
            if d.transportation != "-":
                filled += 1
        return float(filled) / float(total) if total else 0.0


def _is_persona_action(action: Dict[str, Any]) -> bool:
    return action.get("type") in {
        "set_accommodation",
        "set_breakfast",
        "set_lunch",
        "set_dinner",
        "add_attraction1",
        "add_attraction2",
    }


class PersonaGuidance(GuidanceModel):
    def __init__(self, cfg: Optional[GuidanceConfig] = None):
        self.cfg = cfg
        self._pref_cache: Dict[str, Dict[str, str]] = {}
        self._pref_client = None

    def _get_llm_preferences(self, env: TripCraftEnv) -> Dict[str, str]:
        cfg = self.cfg
        persona = (getattr(env.row, "persona", None) or "").strip()
        if not cfg or not persona or not (cfg.model or cfg.endpoint):
            return {}
        cache_key = _safe_json_dump(
            {
                "persona": persona,
                "local_constraint": getattr(env.row, "local_constraint", None),
                "required_cuisines": getattr(env, "required_cuisines", []),
                "required_attraction_types": getattr(env, "required_attraction_types", []),
            }
        )
        if cache_key in self._pref_cache:
            return self._pref_cache[cache_key]
        prefs: Dict[str, str] = {}
        try:
            if cfg.model:
                if self._pref_client is None:
                    from .ollama_client import OllamaClient

                    base_url = (cfg.base_url or "http://localhost:11434").rstrip("/")
                    self._pref_client = OllamaClient(
                        base_url=base_url,
                        model=cfg.model,
                        timeout_sec=cfg.timeout_sec,
                    )
                prompt = _render_llm_pref_prompt(persona, env)
                prefs = _normalize_llm_prefs(self._pref_client.generate_json(prompt))
            elif cfg.endpoint:
                import requests

                payload = {
                    "type": "persona_preferences",
                    "persona": persona,
                    "local_constraint": getattr(env.row, "local_constraint", None),
                    "required_cuisines": getattr(env, "required_cuisines", []),
                    "required_attraction_types": getattr(env, "required_attraction_types", []),
                }
                resp = requests.post(cfg.endpoint, json=payload, timeout=cfg.timeout_sec)
                resp.raise_for_status()
                prefs = _normalize_llm_prefs(resp.json())
        except Exception:
            prefs = {}
        self._pref_cache[cache_key] = prefs
        return prefs

    def prior(self, state: State, actions: List[Dict[str, Any]], env: TripCraftEnv) -> List[float]:
        if not actions:
            return []
        persona_embeddings = get_persona_embeddings(env.row.persona or "")
        if not persona_embeddings:
            return [1.0 / len(actions)] * len(actions)

        llm_prefs = self._get_llm_preferences(env)
        food_pref = llm_prefs.get("food_preferences", "")
        attraction_pref = llm_prefs.get("attraction_preferences", "")

        scores: List[float] = []
        for action in actions:
            if not _is_persona_action(action):
                scores.append(0.0)
                continue
            name = action.get("eval_poi_name") or action.get("name") or ""
            if not name or name == "-":
                scores.append(0.0)
                continue
            action_type = action.get("type") or ""
            kind = "stay" if action_type == "set_accommodation" else "visit"

            name_score: Optional[float] = None
            poi_name = candidate_poi_name(name, kind)
            if poi_name:
                poi_emb = get_poi_embedding(poi_name)
                if poi_emb is not None:
                    name_score = average_persona_similarity(poi_emb, persona_embeddings)

            extra_score: Optional[float] = None
            if action_type in {"set_breakfast", "set_lunch", "set_dinner"}:
                meta = action.get("meta") or {}
                features = _normalize_list(meta.get("features"))
                city = _action_city(action, state, env)
                csv_features = _lookup_restaurant_features(name, city)
                if csv_features:
                    features = _merge_unique(features, csv_features) if features else csv_features
                extra_text = _list_to_text(features)
                extra_score = _persona_text_similarity(extra_text, persona_embeddings)
                llm_score = text_similarity(extra_text, food_pref)
                extra_score = _blend_scores(extra_score, llm_score, _LLM_PREF_WEIGHT)
            elif action_type in {"add_attraction1", "add_attraction2"}:
                meta = action.get("meta") or {}
                subcats = _normalize_list(meta.get("subcategories"))
                city = _action_city(action, state, env)
                csv_subcats = _lookup_attraction_subcategories(name, city)
                if csv_subcats:
                    subcats = _merge_unique(subcats, csv_subcats) if subcats else csv_subcats
                extra_text = _list_to_text(subcats)
                extra_score = _persona_text_similarity(extra_text, persona_embeddings)
                llm_score = text_similarity(extra_text, attraction_pref)
                extra_score = _blend_scores(extra_score, llm_score, _LLM_PREF_WEIGHT)

            scores.append(_combine_scores(name_score, extra_score))

        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        if total <= 0:
            return [1.0 / len(actions)] * len(actions)
        return [v / total for v in exp_scores]

    def value(self, state: State, env: TripCraftEnv) -> float:
        return 0.0


class LLMEndpointGuidance(GuidanceModel):
    def __init__(self, endpoint: str, timeout_sec: float = 10.0):
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec

    def prior(self, state: State, actions: List[Dict[str, Any]], env: TripCraftEnv) -> List[float]:
        if not actions:
            return []
        import requests
        payload = {
            "type": "prior",
            "persona": env.row.persona,
            "local_constraint": env.row.local_constraint,
            "state": {
                "day": state.day,
                "current_city": state.drafts[state.day - 1].current_city,
            },
            "actions": actions,
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        priors = data.get("priors")
        if not isinstance(priors, list) or len(priors) != len(actions):
            return [1.0 / len(actions)] * len(actions)
        total = sum(float(x) for x in priors if x is not None)
        if total <= 0:
            return [1.0 / len(actions)] * len(actions)
        return [float(x) / total for x in priors]

    def value(self, state: State, env: TripCraftEnv) -> float:
        import requests
        payload = {
            "type": "value",
            "persona": env.row.persona,
            "local_constraint": env.row.local_constraint,
            "state": {
                "day": state.day,
                "drafts": [
                    {
                        "breakfast": d.breakfast,
                        "lunch": d.lunch,
                        "dinner": d.dinner,
                        "accommodation": d.accommodation,
                        "attractions": d.attractions,
                    }
                    for d in state.drafts
                ],
            },
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        try:
            return float(data.get("value", 0.0))
        except Exception:
            return 0.0


class OllamaGuidance(GuidanceModel):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_sec: float = 20.0,
        prior_prompt_path: Optional[str] = None,
        value_prompt_path: Optional[str] = None,
    ):
        from .ollama_client import OllamaClient

        self.client = OllamaClient(
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
            prior_prompt_path=prior_prompt_path,
            value_prompt_path=value_prompt_path,
        )

    def _state_summary(self, state: State, env: TripCraftEnv) -> Dict[str, Any]:
        draft = state.drafts[state.day - 1]
        used_pois: List[str] = []
        for d in state.drafts:
            for key in ("breakfast", "lunch", "dinner", "accommodation"):
                val = getattr(d, key, "-")
                if val and val != "-":
                    used_pois.append(val)
            for attr in d.attractions:
                if attr and attr != "-":
                    used_pois.append(attr)
        return {
            "day": state.day,
            "current_city": draft.current_city,
            "persona": env.row.persona,
            "local_constraint": env.row.local_constraint,
            "people_number": env.row.people_number,
            "budget": env.row.budget,
            "required_cuisines": env.required_cuisines,
            "required_attraction_types": env.required_attraction_types,
            "used_pois": sorted(set(used_pois)),
            "draft": {
                "breakfast": draft.breakfast,
                "lunch": draft.lunch,
                "dinner": draft.dinner,
                "accommodation": draft.accommodation,
                "attractions": draft.attractions,
            },
        }

    def prior(self, state: State, actions: List[Dict[str, Any]], env: TripCraftEnv) -> List[float]:
        summary = self._state_summary(state, env)
        return self.client.get_prior(summary, actions)

    def value(self, state: State, env: TripCraftEnv) -> float:
        summary = self._state_summary(state, env)
        return self.client.get_value(summary)


def build_guidance(cfg: GuidanceConfig) -> GuidanceModel:
    if cfg.mode == "heuristic":
        return HeuristicGuidance()
    if cfg.mode == "persona":
        return PersonaGuidance(cfg)
    if cfg.mode == "llm":
        endpoint = cfg.endpoint or os.environ.get("TRIPCRAFT_LLM_ENDPOINT")
        if not endpoint:
            raise ValueError("LLM guidance requires --guidance_endpoint or TRIPCRAFT_LLM_ENDPOINT.")
        return LLMEndpointGuidance(endpoint=endpoint, timeout_sec=cfg.timeout_sec)
    if cfg.mode == "ollama":
        base_url = cfg.base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = cfg.model or os.environ.get("OLLAMA_MODEL")
        if not model:
            raise ValueError("Ollama guidance requires --llm_model, llm_config:model, or OLLAMA_MODEL.")
        return OllamaGuidance(
            base_url=base_url,
            model=model,
            timeout_sec=cfg.timeout_sec,
            prior_prompt_path=cfg.prior_prompt_path,
            value_prompt_path=cfg.value_prompt_path,
        )
    return NullGuidance()
