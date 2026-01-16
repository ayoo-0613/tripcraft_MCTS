from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .env import State, TripCraftEnv
from .persona_utils import average_persona_similarity, candidate_poi_name, get_persona_embeddings, get_poi_embedding


@dataclass
class GuidanceConfig:
    mode: str = "none"  # none | heuristic | persona | llm | ollama
    endpoint: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    prior_prompt_path: Optional[str] = None
    value_prompt_path: Optional[str] = None
    timeout_sec: float = 10.0


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
    def prior(self, state: State, actions: List[Dict[str, Any]], env: TripCraftEnv) -> List[float]:
        if not actions:
            return []
        persona_embeddings = get_persona_embeddings(env.row.persona or "")
        if not persona_embeddings:
            return [1.0 / len(actions)] * len(actions)

        scores: List[float] = []
        for action in actions:
            if not _is_persona_action(action):
                scores.append(0.0)
                continue
            name = action.get("eval_poi_name") or action.get("name") or ""
            if not name or name == "-":
                scores.append(0.0)
                continue
            kind = "stay" if action.get("type") == "set_accommodation" else "visit"
            poi_name = candidate_poi_name(name, kind)
            if not poi_name:
                scores.append(0.0)
                continue
            poi_emb = get_poi_embedding(poi_name)
            if poi_emb is None:
                scores.append(0.0)
                continue
            scores.append(average_persona_similarity(poi_emb, persona_embeddings))

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
        return PersonaGuidance()
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
