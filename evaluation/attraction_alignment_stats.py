import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import os
import sys

# Allow running from within `evaluation/` (repo scripts follow this pattern).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.paths import tripcraft_db_root


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_payload(obj: Dict[str, any]) -> Dict[str, any]:
    data = obj.get("JSON")
    if isinstance(data, dict):
        payload: Dict[str, any] = {}
        payload.update(data)
        payload.update({k: v for k, v in obj.items() if k not in {"JSON"}})
        return payload
    return obj


def _get_mu_d_type(attraction: str, city: str, df: pd.DataFrame) -> Optional[float]:
    attraction_norm = str(attraction or "").strip().lower()
    city_norm = str(city or "").strip().lower()
    if not attraction_norm:
        return None
    if city_norm:
        match = df[
            (df["City"].astype(str).str.strip().str.lower() == city_norm)
            & (df["name"].astype(str).str.strip().str.lower() == attraction_norm)
        ]
    else:
        match = df[df["name"].astype(str).str.strip().str.lower() == attraction_norm]
    if not match.empty:
        try:
            return float(match.iloc[0]["visit_duration"])
        except Exception:
            return None
    return None


def _parse_duration_hours(poi: str) -> Optional[float]:
    try:
        time_info = poi.split("from")[1].split("to")
        start_time = time_info[0].strip()
        end_time = time_info[1].split(",")[0].strip()
    except Exception:
        try:
            time_info = poi.rsplit("from", 1)[1].split("to")
            start_time = time_info[0].strip()
            end_time = time_info[1].split(",")[0].strip()
        except Exception:
            return None
    try:
        start_hour = int(start_time.split(":")[0]) + int(start_time.split(":")[1]) / 60
        end_hour = int(end_time.split(":")[0]) + int(end_time.split(":")[1]) / 60
    except Exception:
        return None
    return end_hour - start_hour


def _mu_d(mu_d_type: float, persona: str, num_attractions: int) -> float:
    k = 16.61 / 60
    mu_d_max = 4
    mu_d_min = 0
    if "Adventure Seeker" in (persona or ""):
        return mu_d_type - k * (num_attractions - mu_d_min)
    return mu_d_type + k * (mu_d_max - num_attractions)


def _summarize_errors(errors: List[float]) -> Dict[str, float]:
    if not errors:
        return {}
    arr = np.array(errors, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(tripcraft_db_root() / "attraction" / "cleaned_attractions_final.csv")

    total_attractions = 0
    matched_mu = 0
    matched_duration = 0
    errors: List[float] = []

    for i, obj in enumerate(_iter_jsonl(args.gen_file), start=1):
        if args.max_samples and i > args.max_samples:
            break
        payload = _get_payload(obj)
        persona = payload.get("persona") or ""
        for day in payload.get("plan", []):
            attraction_field = day.get("attraction", "") or ""
            if not attraction_field or attraction_field.strip() == "-":
                continue
            attractions = [a.strip() for a in attraction_field.split(";") if a.strip()]
            num_attractions = len(attractions)
            poi_list = day.get("point_of_interest_list", "") or ""
            pois = [p.strip() for p in poi_list.split(";") if p.strip()]

            for attr in attractions:
                total_attractions += 1
                if "," in attr:
                    name, city = attr.rsplit(",", 1)
                    name = name.strip()
                    city = city.strip()
                else:
                    name = attr.strip()
                    city = ""

                mu_d_type = _get_mu_d_type(name, city, df)
                if mu_d_type is not None:
                    matched_mu += 1
                else:
                    continue

                duration = None
                for poi in pois:
                    if name.strip() in poi and name.strip() != "-":
                        duration = _parse_duration_hours(poi)
                        break
                if duration is None:
                    continue
                matched_duration += 1
                mu_d_val = _mu_d(mu_d_type, persona, num_attractions)
                errors.append(duration - mu_d_val)

    match_rate = (matched_mu / total_attractions) if total_attractions else 0.0
    duration_rate = (matched_duration / matched_mu) if matched_mu else 0.0

    print(f"total_attractions: {total_attractions}")
    print(f"matched_mu_d_type: {matched_mu}")
    print(f"mu_d_type_match_rate: {match_rate}")
    print(f"duration_matched: {matched_duration}")
    print(f"duration_match_rate: {duration_rate}")
    stats = _summarize_errors(errors)
    if stats:
        print(
            "duration_minus_mu_d: "
            f"mean={stats['mean']}, mean_abs={stats['mean_abs']}, "
            f"median={stats['median']}, p10={stats['p10']}, p90={stats['p90']}"
        )


if __name__ == "__main__":
    main()
