#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import sys
from pathlib import Path


def _normalize_key(value):
    return " ".join(str(value or "").strip().lower().split())


def _clean_tag(tag):
    tag = str(tag or "").strip().strip("\"'")
    return " ".join(tag.split())


def _parse_tag_list(raw):
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text in {"-", "nan", "none", "None"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [_clean_tag(item) for item in parsed if _clean_tag(item)]
        except (ValueError, SyntaxError):
            pass
        text = text[1:-1]
    for delim in [";", "|", ","]:
        if delim in text:
            parts = [_clean_tag(part) for part in text.split(delim)]
            return [part for part in parts if part and part != "-"]
    cleaned = _clean_tag(text)
    return [cleaned] if cleaned and cleaned != "-" else []


def _merge_tags(existing, new_tags):
    seen = {_normalize_key(tag) for tag in existing}
    for tag in new_tags:
        key = _normalize_key(tag)
        if not key or key in seen:
            continue
        existing.append(tag)
        seen.add(key)
    return existing


def _load_tag_map(csv_path, name_field, city_field, tags_field):
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = _normalize_key(row.get(name_field))
            city = _normalize_key(row.get(city_field))
            if not name or not city:
                continue
            tags = _parse_tag_list(row.get(tags_field))
            if not tags:
                continue
            key = (city, name)
            mapping[key] = _merge_tags(mapping.get(key, []), tags)
    return mapping


def _build_name_index(mapping):
    counts = {}
    name_tags = {}
    for (city, name), tags in mapping.items():
        counts[name] = counts.get(name, 0) + 1
        name_tags.setdefault(name, tags)
    return {name: tags for name, tags in name_tags.items() if counts.get(name) == 1}


def _split_items(value):
    items = []
    for chunk in str(value or "").split(";"):
        chunk = chunk.strip()
        if not chunk or chunk == "-":
            continue
        items.append(chunk)
    return items


def _split_name_city(value):
    if "," in value:
        name, city = value.rsplit(",", 1)
        return name.strip(), city.strip()
    return value.strip(), ""


def _extract_city_from_field(value):
    for item in _split_items(value):
        _, city = _split_name_city(item)
        if city:
            return city
    return ""


def _extract_day_city(day):
    current_city = str(day.get("current_city", "") or "").strip()
    if " to " in current_city:
        city = current_city.split(" to ", 1)[1].strip()
        if city:
            return city
    for field in ["breakfast", "lunch", "dinner", "accommodation", "attraction", "event"]:
        city = _extract_city_from_field(day.get(field, ""))
        if city:
            return city
    return ""


def _collect_names(day, fields):
    names = set()
    for field in fields:
        for item in _split_items(day.get(field, "")):
            name, _ = _split_name_city(item)
            if name:
                names.add(_normalize_key(name))
    return names


def _inject_attrs(raw, tags):
    if not tags or "attrs:" in raw or "," not in raw:
        return raw
    name, rest = raw.split(",", 1)
    rest = rest.lstrip()
    tag_str = " ".join(tags)
    return f"{name}, attrs: {tag_str}, {rest}"


def _get_tags_for_poi(
    name_norm,
    city_norm,
    restaurant_names,
    attraction_names,
    rest_map,
    rest_name_index,
    attr_map,
    attr_name_index,
    max_tags,
):
    category = None
    if name_norm in restaurant_names:
        category = "restaurant"
    elif name_norm in attraction_names:
        category = "attraction"
    else:
        if (city_norm, name_norm) in rest_map:
            category = "restaurant"
        elif (city_norm, name_norm) in attr_map:
            category = "attraction"

    if category == "restaurant":
        tags = rest_map.get((city_norm, name_norm)) or rest_name_index.get(name_norm, [])
    elif category == "attraction":
        tags = attr_map.get((city_norm, name_norm)) or attr_name_index.get(name_norm, [])
    else:
        tags = []

    return tags[:max_tags] if tags else []


def _process_plan(plan, rest_map, rest_name_index, attr_map, attr_name_index, max_tags):
    changed = False
    if not isinstance(plan, list):
        return changed
    for day in plan:
        if not isinstance(day, dict):
            continue
        poi_list = day.get("point_of_interest_list")
        if not poi_list or poi_list == "-":
            continue
        city = _extract_day_city(day)
        city_norm = _normalize_key(city)
        restaurant_names = _collect_names(day, ["breakfast", "lunch", "dinner"])
        attraction_names = _collect_names(day, ["attraction"])
        new_parts = []
        day_changed = False
        for part in str(poi_list).split(";"):
            raw = part.strip()
            if not raw:
                continue
            updated = raw
            if (" visit " in raw or " stay " in raw) and "attrs:" not in raw:
                name = raw.split(",", 1)[0].strip()
                name_norm = _normalize_key(name)
                tags = _get_tags_for_poi(
                    name_norm,
                    city_norm,
                    restaurant_names,
                    attraction_names,
                    rest_map,
                    rest_name_index,
                    attr_map,
                    attr_name_index,
                    max_tags,
                )
                if tags:
                    updated = _inject_attrs(raw, tags)
            new_parts.append(updated)
            if updated != raw:
                day_changed = True
        if day_changed:
            day["point_of_interest_list"] = "; ".join(new_parts)
            changed = True
    return changed


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_restaurants = repo_root / "TripCraft/TripCraft_database/restaurants/cleaned_restaurant_details_2024.csv"
    default_attractions = repo_root / "TripCraft/TripCraft_database/attraction/cleaned_attractions_final.csv"

    parser = argparse.ArgumentParser(description="Inject attrs tags into plan point_of_interest_list entries.")
    parser.add_argument("--input", required=True, help="Input JSONL file path (use - for stdin).")
    parser.add_argument("--output", required=True, help="Output JSONL file path (use - for stdout).")
    parser.add_argument("--restaurants-csv", default=str(default_restaurants))
    parser.add_argument("--attractions-csv", default=str(default_attractions))
    parser.add_argument("--max-tags", type=int, default=3)
    args = parser.parse_args()

    rest_map = _load_tag_map(args.restaurants_csv, "name", "City", "features")
    attr_map = _load_tag_map(args.attractions_csv, "name", "City", "subcategories")
    rest_name_index = _build_name_index(rest_map)
    attr_name_index = _build_name_index(attr_map)

    in_handle = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    out_handle = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        for line in in_handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            _process_plan(obj.get("plan"), rest_map, rest_name_index, attr_map, attr_name_index, args.max_tags)
            out_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if in_handle is not sys.stdin:
            in_handle.close()
        if out_handle is not sys.stdout:
            out_handle.close()


if __name__ == "__main__":
    main()
