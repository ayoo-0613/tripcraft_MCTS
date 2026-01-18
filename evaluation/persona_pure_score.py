#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from qualitative_metrics import _get_bert, get_bert_embedding


def compute_persona_score_pure(travel_plan, bert_model, bert_tokenizer):
    """Compute persona score using only the raw POI name (up to first comma)."""
    persona = travel_plan["persona"]
    persona_components = {
        "Traveler Type": None,
        "Purpose of Travel": None,
        "Spending Preference": None,
        "Location Preference": None,
    }

    for key in persona_components.keys():
        start_idx = persona.find(key + ":") + len(key) + 1
        end_idx = persona.find(";", start_idx)
        if end_idx == -1:
            end_idx = len(persona)
        persona_components[key] = persona[start_idx:end_idx].strip()

    persona_embeddings = {
        key: get_bert_embedding(value, bert_tokenizer, bert_model)
        for key, value in persona_components.items()
    }

    total_score = 0
    total_pois = 0

    for day in travel_plan["plan"]:
        poi_list = day.get("point_of_interest_list", "").split(";")
        for poi in poi_list:
            poi = poi.strip()
            if not poi:
                continue
            poi_name = poi.split(",", 1)[0].strip()
            if not poi_name:
                continue
            poi_embedding = get_bert_embedding(poi_name, bert_tokenizer, bert_model)

            poi_score = 0
            for _, persona_embedding in persona_embeddings.items():
                poi_score += cosine_similarity([persona_embedding], [poi_embedding])[0][0]
            poi_score /= len(persona_components)

            total_score += poi_score
            total_pois += 1

    return total_score / total_pois if total_pois > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", required=True, help="Path to generated jsonl plan file.")
    parser.add_argument("--max_plans", type=int, default=0, help="If >0, only evaluate first N plans.")
    parser.add_argument(
        "--skip_zero_days",
        action="store_true",
        help="Skip plans where plan[0]['days'] == 0 (mirrors evaluation script).",
    )
    args = parser.parse_args()

    gen_path = Path(args.gen_file)
    if not gen_path.exists():
        raise SystemExit(f"File not found: {gen_path}")

    tokenizer, model = _get_bert()
    if tokenizer is None or model is None:
        raise SystemExit("BERT model not available; ensure bert-base-uncased is cached locally.")

    plans_used = 0
    plans_scored = 0
    total_plan_score = 0.0

    with gen_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if args.max_plans and idx > args.max_plans:
                break
            line = line.strip()
            if not line:
                continue
            plan = json.loads(line)
            plans_used += 1

            if args.skip_zero_days and plan.get("plan") and plan["plan"][0].get("days") == 0:
                continue

            score = compute_persona_score_pure(plan, model, tokenizer)
            total_plan_score += score
            plans_scored += 1

    avg_score = total_plan_score / plans_scored if plans_scored > 0 else 0
    print(f"plans_used={plans_used} plans_scored={plans_scored}")
    print(f"avg_persona_pure={avg_score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
