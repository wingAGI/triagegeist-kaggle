#!/usr/bin/env python
"""Regenerate a submission from saved full models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triagegeist.config import ARTIFACTS_DIR, ID_COLUMN, TARGET_COLUMN  # noqa: E402
from triagegeist.data import load_merged  # noqa: E402
from triagegeist.features import add_engineered_features  # noqa: E402
from triagegeist.modeling import predict_with_full_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ARTIFACTS_DIR / "models" / "ensemble.joblib",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ARTIFACTS_DIR / "submission.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = joblib.load(args.model_path)
    test_df = add_engineered_features(load_merged("test")).reset_index(drop=True)
    predictions = predict_with_full_models(models, test_df)
    submission = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            TARGET_COLUMN: predictions,
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)
    print(f"wrote submission to {args.output_path}")


if __name__ == "__main__":
    main()
