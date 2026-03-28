#!/usr/bin/env python
"""Backward-compatible entry point for training reports only."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--structured-weight", type=float, default=0.8)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_pipeline.py"),
        "--folds",
        str(args.folds),
        "--structured-weight",
        str(args.structured_weight),
        "--skip-final-fit",
    ]
    if args.sample_size:
        cmd.extend(["--sample-size", str(args.sample_size)])
    if args.skip_figures:
        cmd.append("--skip-figures")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
