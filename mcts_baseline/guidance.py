from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .env import State, TripCraftEnv


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


_BERT_TOKENIZER = None
_BERT_MODEL = None
_BERT_TRIED = False


def _get_bert():
    global _BERT_TOKENIZER, _BERT_MODEL, _BERT_TRIED
    if _BERT_TRIED:
        return _BERT_TOKENIZER, _BERT_MODEL
    _BERT_TRIED = True
    try:
        from transformers import BertTokenizer, BertModel

        try:
            _BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            _BERT_MODEL = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        except TypeError:
            _BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
            _BERT_MODEL = BertModel.from_pretrained("bert-base-uncased")
        if _BERT_MODEL is not None:
            _BERT_MODEL.eval()
    except Exception:
        _BERT_TOKENIZER = None
        _BERT_MODEL = None
    return _BERT_TOKENIZER, _BERT_MODEL


def _get_bert_embedding(text: str, tokenizer: Any, model: Any):
    if tokenizer is None or model is None:
        return None
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()


def _extract_persona_components(persona: str) -> Dict[str, str]:
    components = {
        "Traveler Type": None,
        "Purpose of Travel": None,
        "Spending Preference": None,
        "Location Preference": None,
    }
    for key in components.keys():
        start_idx = persona.find(key + ":") + len(key) + 1
        end_idx = persona.find(";", start_idx)
        if end_idx == -1:
            end_idx = len(persona)
        components[key] = persona[start_idx:end_idx].strip()
    return components


def _extract_poi_name(poi: str) -> str:
    # Match the evaluator logic exactly to avoid mismatches.
    if "stay" in poi:
        return poi.split("stay")[0].strip()[:-1]
    return poi.split("visit")[0].strip()[:-1]


def _action_poi_name(action: Dict[str, Any]) -> str:
    name = action.get("eval_poi_name") or action.get("name") or ""
    if not name or name == "-":
        return ""
    kind = "stay" if action.get("type") == "set_accommodation" else "visit"
    return _extract_poi_name(f"{name}, {kind}")


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
    def __init__(self) -> None:
        self._persona_text: Optional[str] = None
        self._persona_embeddings: Optional[Dict[str, Any]] = None
        self._poi_embeddings: Dict[str, Any] = {}

    def _ensure_persona_embeddings(self, persona: str) -> bool:
        if self._persona_embeddings is not None and persona == self._persona_text:
            return True
        tokenizer, model = _get_bert()
        if tokenizer is None or model is None:
            return False
        components = _extract_persona_components(persona or "")
        embeddings: Dict[str, Any] = {}
        for key, value in components.items():
            emb = _get_bert_embedding(value, tokenizer, model)
            if emb is None:
                return False
            embeddings[key] = emb
        self._persona_text = persona
        self._persona_embeddings = embeddings
        return True

    def prior(self, state: State, actions: List[Dict[str, Any]], env: TripCraftEnv) -> List[float]:
        if not actions:
            return []
        tokenizer, model = _get_bert()
        if tokenizer is None or model is None:
            return [1.0 / len(actions)] * len(actions)
        if not self._ensure_persona_embeddings(env.row.persona or ""):
            return [1.0 / len(actions)] * len(actions)

        from sklearn.metrics.pairwise import cosine_similarity

        scores: List[float] = []
        for action in actions:
            if not _is_persona_action(action):
                scores.append(0.0)
                continue
            poi_name = _action_poi_name(action)
            if not poi_name:
                scores.append(0.0)
                continue
            poi_emb = self._poi_embeddings.get(poi_name)
            if poi_emb is None:
                poi_emb = _get_bert_embedding(poi_name, tokenizer, model)
                if poi_emb is None:
                    scores.append(0.0)
                    continue
                self._poi_embeddings[poi_name] = poi_emb
            total = 0.0
            for persona_emb in self._persona_embeddings.values():
                total += cosine_similarity([persona_emb], [poi_emb])[0][0]
            scores.append(total / float(len(self._persona_embeddings)))

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
