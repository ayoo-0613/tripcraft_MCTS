from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .env import TripCraftEnv
from .guidance import GuidanceConfig, build_guidance
from .io import load_rows, load_rows_from_json, row_from_dict
from .mcts import mcts_search
from .ollama_client import OllamaClient
from .ref_parser import build_unified_kb
from .templater import make_output_record_template
from .formatter import fill_template_with_state
from .validate import validate_record


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _load_prompt(path: Optional[str], default_path: Path) -> str:
    return _read_text(Path(path)) if path else _read_text(default_path)


def _render_prompt(template: str, **kwargs: str) -> str:
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def _load_llm_config(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    text = _read_text(path) if path.exists() else value
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _cfg_get(cfg: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in cfg:
            return cfg.get(key)
    return None


def _print_progress(done: int, total: int, width: int = 30) -> None:
    if total <= 0 or not sys.stderr.isatty():
        return
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    suffix = "\n" if done >= total else "\r"
    sys.stderr.write(f"[{bar}] {done}/{total}" + suffix)
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_csv", type=str)
    input_group.add_argument("--input_json", type=str)
    input_group.add_argument("--input_query", type=str)
    input_group.add_argument("--input_query_file", type=str)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--debug_dir", type=str, default=None)
    parser.add_argument("--llm_config", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_base_url", type=str, default=None)
    parser.add_argument("--query_prompt", type=str, default=None)
    parser.add_argument("--query_output_json", type=str, default=None)
    parser.add_argument("--query_context_json", type=str, default=None)
    parser.add_argument("--query_model", type=str, default=None)
    parser.add_argument("--query_base_url", type=str, default=None)
    parser.add_argument("--query_timeout", type=float, default=None)
    parser.add_argument("--guidance", type=str, default="none", choices=["none", "heuristic", "persona", "llm", "ollama"])
    parser.add_argument("--guidance_endpoint", type=str, default=None)
    parser.add_argument("--guidance_model", type=str, default=None)
    parser.add_argument("--guidance_base_url", type=str, default=None)
    parser.add_argument("--guidance_prior_prompt", type=str, default=None)
    parser.add_argument("--guidance_value_prompt", type=str, default=None)
    parser.add_argument("--guidance_timeout", type=float, default=None)
    parser.add_argument("--guidance_value_weight", type=float, default=0.0)
    parser.add_argument("--guidance_prior_c", type=float, default=1.4)
    parser.add_argument("--temporal_guidance", type=str, default="none", choices=["none", "ollama"])
    parser.add_argument("--temporal_prompt", type=str, default=None)
    parser.add_argument("--temporal_timeout", type=float, default=None)
    args = parser.parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be a positive integer.")
    if not (0.0 <= args.guidance_value_weight <= 1.0):
        raise ValueError("--guidance_value_weight must be in [0, 1].")

    llm_cfg = _load_llm_config(args.llm_config)
    llm_model = args.llm_model or _cfg_get(llm_cfg, "model")
    if llm_model is None:
        llm_model = args.guidance_model or args.query_model or _cfg_get(llm_cfg, "guidance_model", "query_model")
    llm_base_url = args.llm_base_url or _cfg_get(llm_cfg, "base_url")
    if llm_base_url is None:
        llm_base_url = args.guidance_base_url or args.query_base_url or _cfg_get(llm_cfg, "guidance_base_url", "query_base_url")
    if llm_base_url is None:
        llm_base_url = "http://localhost:11434"

    query_model = llm_model
    query_base_url = llm_base_url
    query_timeout = args.query_timeout
    if query_timeout is None:
        query_timeout = _cfg_get(llm_cfg, "timeout_sec", "query_timeout_sec")
    query_timeout = float(query_timeout) if query_timeout is not None else 20.0
    query_prompt_path = args.query_prompt or _cfg_get(llm_cfg, "query_prompt")

    guidance_endpoint = args.guidance_endpoint or _cfg_get(llm_cfg, "endpoint", "guidance_endpoint")
    guidance_model = llm_model
    guidance_base_url = llm_base_url
    guidance_prior_prompt = args.guidance_prior_prompt or _cfg_get(llm_cfg, "prior_prompt", "guidance_prior_prompt")
    guidance_value_prompt = args.guidance_value_prompt or _cfg_get(llm_cfg, "value_prompt", "guidance_value_prompt")
    guidance_timeout = args.guidance_timeout
    if guidance_timeout is None:
        guidance_timeout = _cfg_get(llm_cfg, "timeout_sec", "guidance_timeout_sec")
    guidance_timeout = float(guidance_timeout) if guidance_timeout is not None else 10.0

    temporal_model = llm_model
    temporal_base_url = llm_base_url
    temporal_prompt = args.temporal_prompt or _cfg_get(llm_cfg, "temporal_prompt")
    temporal_timeout = args.temporal_timeout
    if temporal_timeout is None:
        temporal_timeout = _cfg_get(llm_cfg, "temporal_timeout_sec", "temporal_timeout")
    temporal_timeout = float(temporal_timeout) if temporal_timeout is not None else guidance_timeout

    rows = []
    if args.input_csv:
        rows = load_rows(args.input_csv)
    elif args.input_json:
        rows = load_rows_from_json(args.input_json)
    else:
        query_text = args.input_query
        if not query_text and args.input_query_file:
            query_text = _read_text(Path(args.input_query_file))
        if not query_text:
            raise ValueError("Query input is empty.")
        if not query_model:
            raise ValueError("--llm_model or llm_config:model is required when using --input_query or --input_query_file.")

        prompt_path = Path(__file__).parent / "prompts" / "query_to_json.txt"
        prompt_template = _load_prompt(query_prompt_path, prompt_path)
        context_json = "{}"
        if args.query_context_json:
            context_json = _read_text(Path(args.query_context_json)) or "{}"
        prompt_text = _render_prompt(prompt_template, query_text=query_text, context_json=context_json)

        client = OllamaClient(
            base_url=query_base_url,
            model=query_model,
            timeout_sec=query_timeout,
        )
        query_obj = client.generate_json(prompt_text)
        if not query_obj:
            raise ValueError("LLM query-to-JSON output is empty or invalid.")
        if args.query_output_json:
            _write_json(Path(args.query_output_json), query_obj)
        rows = [row_from_dict(query_obj, idx_default=1)]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else (out_path.parent / "debug_mcts_baseline")
    debug_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", errors="replace") as f:
        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            if not row.ref_blocks:
                raise ValueError(f"row idx={row.idx} is missing reference_information.")
            template = make_output_record_template(row)
            kb = build_unified_kb(row.org, row.ref_blocks)
            env = TripCraftEnv(row=row, kb=kb, topk=args.topk)
            guidance = build_guidance(
                GuidanceConfig(
                    mode=args.guidance,
                    endpoint=guidance_endpoint,
                    model=guidance_model,
                    base_url=guidance_base_url,
                    prior_prompt_path=guidance_prior_prompt,
                    value_prompt_path=guidance_value_prompt,
                    timeout_sec=guidance_timeout,
                )
            )
            temporal_client = None
            if args.temporal_guidance == "ollama":
                if not temporal_model:
                    raise ValueError("Temporal guidance requires --llm_model or llm_config:model.")
                temporal_client = OllamaClient(
                    base_url=temporal_base_url,
                    model=temporal_model,
                    timeout_sec=temporal_timeout,
                    temporal_prompt_path=temporal_prompt,
                )
            terminal_state = mcts_search(
                env,
                rollouts=args.rollouts,
                topk=args.topk,
                guidance=guidance,
                prior_c=args.guidance_prior_c,
                value_weight=args.guidance_value_weight,
            )
            rec = fill_template_with_state(template, row, kb, terminal_state, temporal_client=temporal_client)

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
            _print_progress(idx, total)


if __name__ == "__main__":
    main()
