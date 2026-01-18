#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


DEFAULT_PROMPT = """You are a travel persona evaluator.

Task:
Given a Persona and a POI list (names only), score how well the POIs align with the Persona.

Inputs:
Persona:
{persona}

POI list (by day, names only):
{poi_list}

Rules:
1) Use only POI NAMES. Ignore times, durations, prices, distances, transit details, and any other fields.
2) If a POI name is unfamiliar or ambiguous, treat it as "unknown" and do NOT guess details.
3) If the Persona is vague, infer only what is explicitly stated. Do NOT add unstated preferences.
4) Count evidence from POI names. Use common, widely known cues in names only (e.g., "Museum", "Beach", "Outlet", "Fine Dining", "National Park").
5) Be consistent and conservative. Prefer lower scores when evidence is weak or mixed.

Evaluation procedure (do this internally, output JSON only):
A) Extract Persona signals into these four components:
   - Traveler Type (e.g., solo, couple, family, business, backpacker, luxury, adventure)
   - Purpose of Travel (e.g., culture, nature, relaxation, food, entertainment, sports, shopping)
   - Spending Preference (budget, mid-range, luxury)
   - Location Preference (urban, suburban, nature, coast, mountains, theme park, etc.)

B) For each POI name, assign at most TWO tags that are strongly implied by the name.
   If not strongly implied, tag as "unknown".

C) Score each component in [0.0, 1.0] using this guidance:
   - 1.0: strong alignment, most POIs provide direct evidence for the persona signal
   - 0.7: good alignment, clear evidence but with some neutral or unknown POIs
   - 0.5: mixed, comparable evidence for and against, or mostly neutral/unknown
   - 0.3: weak alignment, little supporting evidence, many unknown POIs
   - 0.0: clear mismatch, POIs mostly contradict the persona signal

D) Compute overall score as the average of the four component scores, then apply penalties:
   - Penalty -0.05 to -0.20 when there are strong contradictions
     Example: Persona says budget-focused but POIs clearly indicate luxury/fine dining.
   - Penalty -0.05 when unknown POIs dominate (over half of POIs are unknown).
   Clamp final score to [0.0, 1.0].

Output format:
Return JSON only. No extra text, no markdown.

Schema:
{
  "score": 0.0,
  "components": {
    "Traveler Type": 0.0,
    "Purpose of Travel": 0.0,
    "Spending Preference": 0.0,
    "Location Preference": 0.0
  },
  "evidence": {
    "Traveler Type": ["POI_A", "POI_B"],
    "Purpose of Travel": ["POI_C"],
    "Spending Preference": ["POI_D"],
    "Location Preference": ["POI_E"]
  },
  "unknown_pois": ["POI_X", "POI_Y"],
  "contradictions": [
    {"component": "Spending Preference", "reason": "persona implies budget but POIs suggest luxury", "pois": ["POI_D"]}
  ],
  "notes": "one or two sentences, concise"
}
"""



def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def _parse_score(text: str) -> Tuple[float, Dict[str, Any]]:
    obj: Dict[str, Any] = {}
    try:
        obj = json.loads(_extract_json(text))
    except Exception:
        obj = {}
    score = obj.get("score")
    try:
        score_f = float(score)
    except Exception:
        m = re.search(r"([0-9]*\\.?[0-9]+)", text)
        score_f = float(m.group(1)) if m else 0.0
    score_f = max(0.0, min(1.0, score_f))
    return score_f, obj


def _chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": float(temperature)}
    resp = requests.post(url, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("message") or {}).get("content")
    return str(content or "")


def _iter_poi_names(poi_list_str: str) -> List[str]:
    names = []
    for raw in str(poi_list_str or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if raw.endswith("."):
            raw = raw[:-1]
        name = raw.split(",", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _format_plan_poi_list(plan: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for day in plan.get("plan", []):
        day_idx = day.get("days")
        names = _iter_poi_names(day.get("point_of_interest_list", ""))
        if not names:
            continue
        label = f"Day {day_idx}" if day_idx is not None else "Day"
        chunks.append(f"{label}: " + "; ".join(names))
    return "\n".join(chunks)


def _load_prompt(path: str | None) -> str:
    if not path:
        return DEFAULT_PROMPT
    return Path(path).read_text(encoding="utf-8")


def _score_once(
    *,
    persona: str,
    poi_list: str,
    prompt: str,
    base_url: str,
    model: str,
    timeout_sec: float,
    temperature: float,
) -> Tuple[float, Dict[str, Any], str]:
    content = prompt.format(persona=persona, poi_list=poi_list)
    text = _chat(
        base_url=base_url,
        model=model,
        messages=[{"role": "user", "content": content}],
        timeout_sec=timeout_sec,
        temperature=temperature,
    ).strip()
    score, obj = _parse_score(text)
    return score, obj, text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", type=str, default="", help="Persona string.")
    parser.add_argument("--poi_list", type=str, default="", help="POI list string.")
    parser.add_argument("--gen_file", type=str, default="", help="Path to generated jsonl plan file.")
    # Removed --max_plans argument; always process all lines in the JSONL file.
    parser.add_argument("--prompt_path", type=str, default="", help="Optional prompt template path.")
    parser.add_argument("--model", type=str, default=os.getenv("OLLAMA_MODEL", ""), help="Ollama model name.")
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL.",
    )
    parser.add_argument("--timeout_sec", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--print_per_plan", action="store_true", help="Print per-plan scores.")
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("Missing --model or OLLAMA_MODEL.")

    prompt = _load_prompt(args.prompt_path or None)

    if args.gen_file:
        gen_path = Path(args.gen_file)
        if not gen_path.exists():
            raise SystemExit(f"File not found: {gen_path}")
        total = 0.0
        count = 0
        plans_used = 0
        with gen_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                plan = json.loads(line)
                plans_used += 1
                persona = plan.get("persona") or ""
                poi_list = _format_plan_poi_list(plan)
                score, obj, _ = _score_once(
                    persona=persona,
                    poi_list=poi_list,
                    prompt=prompt,
                    base_url=args.base_url,
                    model=args.model,
                    timeout_sec=args.timeout_sec,
                    temperature=args.temperature,
                )
                total += score
                count += 1
                if args.print_per_plan:
                    payload = {"idx": plan.get("idx"), "score": score, "components": obj.get("components")}
                    print(json.dumps(payload, ensure_ascii=True))
        avg = total / count if count else 0.0
        print(f"plans_used={plans_used} plans_scored={count}")
        print(f"avg_persona_llm={avg:.6f}")
        return 0

    if not args.persona or not args.poi_list:
        raise SystemExit("Provide --persona and --poi_list, or --gen_file.")

    score, obj, _ = _score_once(
        persona=args.persona,
        poi_list=args.poi_list,
        prompt=prompt,
        base_url=args.base_url,
        model=args.model,
        timeout_sec=args.timeout_sec,
        temperature=args.temperature,
    )
    print(f"score={score:.6f}")
    if obj:
        print(json.dumps(obj, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
