from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .env import TripCraftEnv
from .io import load_rows
from .mcts import mcts_search
from .ref_parser import build_unified_kb
from .templater import make_output_record_template
from .formatter import fill_template_with_state
from .validate import validate_record


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--debug_dir", type=str, default=None)
    args = parser.parse_args()
    if args.topk != 5:
        raise ValueError("--topk is fixed to 5 for this baseline (per spec).")

    rows = load_rows(args.input_csv)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else (out_path.parent / "debug_mcts_baseline")
    debug_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", errors="replace") as f:
        for row in rows:
            template = make_output_record_template(row)
            kb = build_unified_kb(row.org, row.ref_blocks)
            env = TripCraftEnv(row=row, kb=kb, topk=args.topk)
            terminal_state = mcts_search(env, rollouts=args.rollouts, topk=args.topk)
            rec = fill_template_with_state(template, row, kb, terminal_state)

            errs = validate_record(rec)
            if errs:
                raise ValueError(f"record idx={row.idx} validation errors: {errs}")

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            kb_summary = {
                "idx": row.idx,
                "stages": [
                    {
                        "city": s.city,
                        "counts": {
                            "attractions": int(len(s.attractions)) if s.attractions is not None else 0,
                            "restaurants": int(len(s.restaurants)) if s.restaurants is not None else 0,
                            "accommodations": int(len(s.accommodations)) if s.accommodations is not None else 0,
                            "poi2transit": int(len(s.poi2transit)) if s.poi2transit is not None else 0,
                            "events": int(len(s.events)) if s.events is not None else 0,
                        },
                    }
                    for s in kb.stages
                ],
                "transports": [{"mode": t.mode, "from": t.frm, "to": t.to, "date": t.date, "raw": t.raw} for t in kb.transports],
            }
            _write_json(debug_dir / f"kb_summary_{row.idx}.json", kb_summary)
            _write_json(
                debug_dir / f"plan_trace_{row.idx}.json",
                {
                    "idx": row.idx,
                    "reward": env.evaluate(terminal_state),
                    "budget_used": terminal_state.budget_used,
                    "action_trace": terminal_state.action_trace,
                    "days": [
                        {
                            "day": i + 1,
                            "current_city": terminal_state.drafts[i].current_city,
                            "transportation": terminal_state.drafts[i].transportation,
                            "accommodation": terminal_state.drafts[i].accommodation,
                            "breakfast": terminal_state.drafts[i].breakfast,
                            "attractions": terminal_state.drafts[i].attractions,
                            "lunch": terminal_state.drafts[i].lunch,
                            "dinner": terminal_state.drafts[i].dinner,
                            "poi_blocks": [
                                {
                                    "name": b.name,
                                    "kind": b.kind,
                                    "start": b.start,
                                    "end": b.end,
                                    "nearest_transit": b.nearest_transit,
                                    "dist_m": b.dist_m,
                                }
                                for b in terminal_state.drafts[i].poi_blocks
                            ],
                        }
                        for i in range(row.days)
                    ],
                },
            )
            _write_json(debug_dir / f"record_{row.idx}.json", rec)


if __name__ == "__main__":
    main()
