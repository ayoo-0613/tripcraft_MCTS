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
        if args.eval_mode:
            cmd.append("--eval_mode")
        if args.emit_smt2_dir:
            smt2_dir = output_dir / f"smt2_{d}day"
            cmd += ["--emit_smt2_dir", str(smt2_dir)]
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
