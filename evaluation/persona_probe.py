#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import qualitative_metrics as qm


KEYS = [
    "Traveler Type",
    "Purpose of Travel",
    "Spending Preference",
    "Location Preference",
]

LABELS = {
    "Traveler Type": "traveler",
    "Purpose of Travel": "purpose",
    "Spending Preference": "spending",
    "Location Preference": "location",
}

VARIANT_KEYS = {
    "base",
    "name+values",
    "name+labels",
    "values_only",
    "labels_only",
    "name+traveler",
    "name+purpose",
    "name+spending",
    "name+location",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "the",
    "and",
    "of",
    "in",
    "at",
    "for",
    "to",
    "on",
    "a",
    "an",
    "by",
    "with",
    "from",
}


def _normalize(value: str) -> str:
    return str(value or "").strip().lower()


def _parse_persona(persona: str) -> dict:
    components = {}
    for key in KEYS:
        start_idx = persona.find(key + ":")
        if start_idx == -1:
            components[key] = ""
            continue
        start_idx += len(key) + 1
        end_idx = persona.find(";", start_idx)
        if end_idx == -1:
            end_idx = len(persona)
        components[key] = persona[start_idx:end_idx].strip()
    return components


def _iter_poi_names(poi_list_str: str):
    for raw in str(poi_list_str or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if raw.endswith("."):
            raw = raw[:-1]
        name = raw.split(",", 1)[0].strip()
        if name:
            yield name


def _day_category_index(day: dict):
    def _strip_city(value: str) -> str:
        if not value or value == "-":
            return ""
        if "," in value:
            value = value.rsplit(",", 1)[0].strip()
        return _normalize(value)

    accommodation = _strip_city(day.get("accommodation", ""))
    meals = set(
        filter(
            None,
            [
                _strip_city(day.get("breakfast", "")),
                _strip_city(day.get("lunch", "")),
                _strip_city(day.get("dinner", "")),
            ],
        )
    )

    attractions = set()
    for item in str(day.get("attraction", "") or "").split(";"):
        item = item.strip()
        if not item or item == "-":
            continue
        if "," in item:
            item = item.rsplit(",", 1)[0].strip()
        attractions.add(_normalize(item))

    return accommodation, meals, attractions


def _categorize(name_norm: str, accommodation: str, meals: set, attractions: set) -> str:
    if name_norm and accommodation and name_norm == accommodation:
        return "accommodation"
    if name_norm and name_norm in meals:
        return "meal"
    if name_norm and name_norm in attractions:
        return "attraction"
    return "other"


def _build_variants(name: str, comps: dict) -> dict:
    values = [comps.get(k) for k in KEYS if comps.get(k)]
    label_parts = []
    for key in KEYS:
        value = comps.get(key) or ""
        if value:
            label_parts.append(f"{LABELS[key]}: {value}")
    tag_labels = " | ".join(label_parts)
    tag_values = " | ".join(values)

    variants = {
        "base": name,
        "name+values": f"{name}, {tag_values}" if tag_values else "",
        "name+labels": f"{name}, {tag_labels}" if tag_labels else "",
        "values_only": tag_values,
        "labels_only": tag_labels,
        "name+traveler": f"{name}, {comps.get('Traveler Type')}" if comps.get("Traveler Type") else "",
        "name+purpose": f"{name}, {comps.get('Purpose of Travel')}" if comps.get("Purpose of Travel") else "",
        "name+spending": f"{name}, {comps.get('Spending Preference')}" if comps.get("Spending Preference") else "",
        "name+location": f"{name}, {comps.get('Location Preference')}" if comps.get("Location Preference") else "",
    }
    return variants


def _tokens(text: str):
    for tok in TOKEN_RE.findall(text or ""):
        tok = tok.lower()
        if len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        yield tok


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe what POI content correlates with higher persona scores.")
    parser.add_argument("--gen_file", required=True, help="Path to generated jsonl plan file.")
    parser.add_argument("--max_plans", type=int, default=0, help="Limit number of plans to analyze (0 = all).")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k items to show for names/tokens.")
    parser.add_argument("--min_name_count", type=int, default=2, help="Min count for POI name stats.")
    parser.add_argument("--min_token_count", type=int, default=5, help="Min count for token stats.")
    parser.add_argument("--persona_top_k", type=int, default=10, help="Top-k persona values to show per component.")
    parser.add_argument("--min_persona_count", type=int, default=5, help="Min count for persona bucket stats.")
    parser.add_argument(
        "--persona_token_top_k",
        type=int,
        default=8,
        help="Top-k tokens by lift to show per persona value.",
    )
    parser.add_argument(
        "--persona_token_min_count",
        type=int,
        default=5,
        help="Min count for persona-token stats.",
    )
    parser.add_argument(
        "--variants",
        default="base,name+values,name+labels,values_only,labels_only,name+traveler,name+purpose,name+spending,name+location",
        help="Comma-separated variant keys to compare.",
    )
    args = parser.parse_args()

    variant_keys = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variant_keys) - VARIANT_KEYS)
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    tokenizer, model = qm._get_bert()
    if tokenizer is None or model is None:
        raise SystemExit("BERT model not available; ensure bert-base-uncased is cached locally.")

    embed_cache = {}

    def embed(text: str):
        if text not in embed_cache:
            embed_cache[text] = qm.get_bert_embedding(text, tokenizer, model)
        return embed_cache[text]

    def cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    gen_path = Path(args.gen_file)
    if not gen_path.exists():
        raise SystemExit(f"File not found: {gen_path}")

    totals = Counter()
    counts = Counter()

    category_totals = Counter()
    category_counts = Counter()

    component_totals = Counter()
    persona_bucket_totals = Counter()
    persona_bucket_counts = Counter()
    persona_token_totals = Counter()
    persona_token_counts = Counter()

    name_totals = Counter()
    name_counts = Counter()
    name_cats = defaultdict(Counter)

    token_totals = Counter()
    token_counts = Counter()

    plans_used = 0
    total_pois = 0

    with gen_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if args.max_plans and idx > args.max_plans:
                break
            line = line.strip()
            if not line:
                continue
            plan = json.loads(line)
            persona = plan.get("persona") or ""
            comps = _parse_persona(persona)
            persona_embeds = {k: embed(comps.get(k, "")) for k in KEYS}

            for day in plan.get("plan", []):
                accommodation, meals, attractions = _day_category_index(day)
                for name in _iter_poi_names(day.get("point_of_interest_list", "")):
                    name_norm = _normalize(name)
                    category = _categorize(name_norm, accommodation, meals, attractions)

                    base_embed = embed(name)
                    base_score = sum(cosine(base_embed, persona_embeds[k]) for k in KEYS) / len(KEYS)

                    totals["base"] += base_score
                    counts["base"] += 1

                    category_totals[category] += base_score
                    category_counts[category] += 1

                    for k in KEYS:
                        component_totals[k] += cosine(base_embed, persona_embeds[k])

                    for k in KEYS:
                        value = comps.get(k) or ""
                        if value:
                            persona_bucket_totals[(k, value)] += base_score
                            persona_bucket_counts[(k, value)] += 1

                    name_totals[name] += base_score
                    name_counts[name] += 1
                    name_cats[name][category] += 1

                    tokens = set(_tokens(name))
                    for tok in tokens:
                        token_totals[tok] += base_score
                        token_counts[tok] += 1

                    if variant_keys:
                        variants = _build_variants(name, comps)
                        for vkey in variant_keys:
                            vtext = variants.get(vkey, "")
                            if not vtext:
                                continue
                            vembed = embed(vtext)
                            vscore = sum(cosine(vembed, persona_embeds[k]) for k in KEYS) / len(KEYS)
                            totals[vkey] += vscore
                            counts[vkey] += 1

                    total_pois += 1

                    for k in KEYS:
                        value = comps.get(k) or ""
                        if not value:
                            continue
                        for tok in tokens:
                            persona_token_totals[(k, value, tok)] += base_score
                            persona_token_counts[(k, value, tok)] += 1
            plans_used += 1

    if total_pois == 0:
        raise SystemExit("No POIs found in the input file.")

    base_avg = totals["base"] / counts["base"]

    print(f"plans_used={plans_used} total_pois={total_pois}")
    print(f"baseline_avg={base_avg:.6f}")

    print("\nby_category")
    for cat in sorted(category_counts):
        avg = category_totals[cat] / category_counts[cat]
        print(f"{cat}\tavg={avg:.6f}\tcount={category_counts[cat]}")

    print("\nby_component")
    for key in KEYS:
        avg = component_totals[key] / total_pois
        print(f"{key}\tavg={avg:.6f}")

    if variant_keys:
        print("\nvariant_comparison")
        rows = []
        for vkey in variant_keys:
            if counts[vkey] == 0:
                continue
            avg = totals[vkey] / counts[vkey]
            rows.append((vkey, avg, avg - base_avg, counts[vkey]))
        rows.sort(key=lambda x: x[1], reverse=True)
        for vkey, avg, delta, cnt in rows:
            print(f"{vkey}\tavg={avg:.6f}\tdelta={delta:+.6f}\tcount={cnt}")

    print("\nby_persona_value")
    for key in KEYS:
        rows = []
        for (bucket_key, value), cnt in persona_bucket_counts.items():
            if bucket_key != key:
                continue
            if cnt < args.min_persona_count:
                continue
            avg = persona_bucket_totals[(bucket_key, value)] / cnt
            rows.append((value, avg, cnt))
        rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for value, avg, cnt in rows[: args.persona_top_k]:
            print(f"{key}\t{value}\tavg={avg:.6f}\tcount={cnt}")

    print("\npersona_token_lift")
    for key in KEYS:
        for (bucket_key, value), cnt in persona_bucket_counts.items():
            if bucket_key != key or cnt < args.min_persona_count:
                continue
            baseline = persona_bucket_totals[(bucket_key, value)] / cnt
            token_rows = []
            for (k, v, tok), tcnt in persona_token_counts.items():
                if k != key or v != value:
                    continue
                if tcnt < args.persona_token_min_count:
                    continue
                avg = persona_token_totals[(k, v, tok)] / tcnt
                token_rows.append((tok, avg, avg - baseline, tcnt))
            token_rows.sort(key=lambda x: (x[2], x[3]), reverse=True)
            for tok, avg, lift, tcnt in token_rows[: args.persona_token_top_k]:
                print(f"{key}\t{value}\t{tok}\tavg={avg:.6f}\tlift={lift:+.6f}\tcount={tcnt}")

    print("\ntop_poi_names")
    name_rows = []
    for name, cnt in name_counts.items():
        if cnt < args.min_name_count:
            continue
        avg = name_totals[name] / cnt
        cats = ",".join(f"{k}:{v}" for k, v in name_cats[name].most_common())
        name_rows.append((name, avg, cnt, cats))
    name_rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    for name, avg, cnt, cats in name_rows[: args.top_k]:
        print(f"{name}\tavg={avg:.6f}\tcount={cnt}\tcats={cats}")

    print("\ntop_tokens")
    token_rows = []
    for tok, cnt in token_counts.items():
        if cnt < args.min_token_count:
            continue
        avg = token_totals[tok] / cnt
        token_rows.append((tok, avg, cnt))
    token_rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    for tok, avg, cnt in token_rows[: args.top_k]:
        print(f"{tok}\tavg={avg:.6f}\tcount={cnt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
