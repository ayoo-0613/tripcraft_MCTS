from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_sec: float = 999.0,
        prior_prompt_path: Optional[str] = None,
        value_prompt_path: Optional[str] = None,
        temporal_prompt_path: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.prior_prompt = _load_template(
            prior_prompt_path,
            default_name="ollama_prior.txt",
        )
        self.value_prompt = _load_template(
            value_prompt_path,
            default_name="ollama_value.txt",
        )
        self.temporal_prompt = _load_template(
            temporal_prompt_path,
            default_name="ollama_temporal.txt",
        )

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        import requests

        url = f"{self.base_url}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": False}
        resp = requests.post(url, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content")
        return str(content or "")

    def get_prior(self, state_summary: Dict[str, Any], actions: List[Dict[str, Any]]) -> List[float]:
        if not actions:
            return []
        content = _render_prompt(
            self.prior_prompt,
            state_json=json.dumps(state_summary, ensure_ascii=True),
            actions_json=json.dumps(actions, ensure_ascii=True),
        )
        prompt = {"role": "user", "content": content}
        text = self._chat([prompt]).strip()
        return _parse_priors(text, len(actions))

    def get_value(self, state_summary: Dict[str, Any]) -> float:
        content = _render_prompt(
            self.value_prompt,
            state_json=json.dumps(state_summary, ensure_ascii=True),
            actions_json="[]",
        )
        prompt = {"role": "user", "content": content}
        text = self._chat([prompt]).strip()
        return _parse_value(text)

    def get_temporal_schedule(self, state_summary: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        if not items:
            return []
        content = _render_prompt(
            self.temporal_prompt,
            state_json=json.dumps(state_summary, ensure_ascii=True),
            items_json=json.dumps(items, ensure_ascii=True),
        )
        prompt = {"role": "user", "content": content}
        text = self._chat([prompt]).strip()
        return _parse_temporal_items(text)

    def generate_json(self, prompt_text: str) -> Dict[str, Any]:
        prompt = {"role": "user", "content": prompt_text}
        text = self._chat([prompt]).strip()
        return _parse_json_obj(text)


def _parse_priors(text: str, n: int) -> List[float]:
    try:
        obj = json.loads(_extract_json(text))
        priors = obj.get("priors")
        if isinstance(priors, list) and len(priors) == n:
            vals = [float(x) for x in priors]
            s = sum(vals)
            if s > 0:
                return [v / s for v in vals]
    except Exception:
        pass
    return [1.0 / n] * n


def _parse_value(text: str) -> float:
    try:
        obj = json.loads(_extract_json(text))
        val = float(obj.get("value", 0.0))
        return max(0.0, min(1.0, val))
    except Exception:
        m = re.search(r"([0-9]*\\.?[0-9]+)", text)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except Exception:
                pass
    return 0.0


def _parse_json_obj(text: str) -> Dict[str, Any]:
    try:
        obj = json.loads(_extract_json(text))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\\{.*\\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def _load_template(path: Optional[str], default_name: str) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    default_path = Path(__file__).parent / "prompts" / default_name
    return default_path.read_text(encoding="utf-8")


def _render_prompt(template: str, **kwargs: str) -> str:
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def _parse_temporal_items(text: str) -> List[Dict[str, str]]:
    obj = _parse_json_obj(text)
    items = obj.get("items")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        start = item.get("start")
        end = item.get("end")
        if not isinstance(name, str) or not isinstance(start, str) or not isinstance(end, str):
            continue
        out.append({"name": name, "start": start, "end": end})
    return out
