#!/usr/bin/env python
"""Run a light ablation sweep for Triagegeist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from triagegeist.config import ARTIFACTS_DIR, SUBGROUP_COLUMNS, TEXT_COLUMN
from triagegeist.data import load_merged
from triagegeist.features import add_engineered_features
from triagegeist.modeling import train_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--folds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = add_engineered_features(load_merged("train")).reset_index(drop=True)
    test_df = add_engineered_features(load_merged("test")).reset_index(drop=True).head(1000)

    variants = {
        "ensemble_080": 0.80,
        "ensemble_090": 0.90,
        "ensemble_070": 0.70,
    }
    results: dict[str, dict[str, float]] = {}

    for name, structured_weight in variants.items():
        artifacts = train_ensemble(
            train_frame=train_df,
            test_frame=test_df,
            subgroup_columns=SUBGROUP_COLUMNS,
            sample_size=args.sample_size,
            folds=args.folds,
            structured_weight=structured_weight,
        )
        results[name] = artifacts.metrics
        print(name, artifacts.metrics)

    output_path = ARTIFACTS_DIR / "ablation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
