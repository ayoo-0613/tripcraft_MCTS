#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Z3 solver for TripCraft 3/5/7-day CSVs and emit JSONL plans.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days",
        type=str,
        default="3,5,7",
        help="Comma-separated list of day counts (default: 3,5,7).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Solver_result/outputs",
        help="Output directory for JSONL files.",
    )
    parser.add_argument("--max_rows", type=int, default=-1, help="Max rows per CSV (default: all).")
    parser.add_argument("--solver_timeout_ms", type=int, default=60000)
    parser.add_argument("--llm_parse_query", action="store_true")
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:11434")
    parser.add_argument("--llm_timeout", type=float, default=20.0)
    parser.add_argument("--llm_query_prompt", type=str, default=None)
    parser.add_argument("--llm_render_plan", action="store_true")
    parser.add_argument("--llm_render_prompt", type=str, default=None)
    parser.add_argument(
        "--emit_smt2_dir",
        type=str,
        default="",
        help="Optional dir to dump SMT2 files (empty to disable).",
    )
    parser.add_argument("--eval_mode", action="store_true", help="Use evaluation-friendly output format.")
    args = parser.parse_args()

    root = _repo_root()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    days = [d.strip() for d in args.days.split(",") if d.strip()]
    for d in days:
        csv_path = root / "TripCraft" / "Tripcraftzip" / f"tripcraft_{d}day.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        out_path = output_dir / f"z3_{d}day.jsonl"
        cmd = [
            "python",
            str(root / "Solver_baseline" / "JSON_to_CODE.py"),
            "--input_csv",
            str(csv_path),
            "--output_jsonl",
            str(out_path),
            "--max_rows",
            str(args.max_rows),
            "--solver_timeout_ms",
            str(args.solver_timeout_ms),
        ]
        if args.llm_parse_query:
            cmd.append("--llm_parse_query")
        if args.llm_model:
            cmd += ["--llm_model", str(args.llm_model)]
        if args.llm_base_url:
            cmd += ["--llm_base_url", str(args.llm_base_url)]
        if args.llm_timeout is not None:
            cmd += ["--llm_timeout", str(args.llm_timeout)]
        if args.llm_query_prompt:
            cmd += ["--llm_query_prompt", str(args.llm_query_prompt)]
        if args.llm_render_plan:
            cmd.append("--llm_render_plan")
        if args.llm_render_prompt:
            cmd += ["--llm_render_prompt", str(args.llm_render_prompt)]
        if args.eval_mode:
            cmd.append("--eval_mode")
        if args.emit_smt2_dir:
            smt2_dir = output_dir / f"smt2_{d}day"
            cmd += ["--emit_smt2_dir", str(smt2_dir)]
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
