#!/usr/bin/env python
"""Fit the final model bundle and export a submission file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from triagegeist.config import ARTIFACTS_DIR, ID_COLUMN, TARGET_COLUMN
from triagegeist.data import load_merged
from triagegeist.features import add_engineered_features
from triagegeist.modeling import fit_full_models, predict_with_full_models, save_models
from triagegeist.reporting import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--structured-weight", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(ARTIFACTS_DIR / "models")

    train_df = add_engineered_features(load_merged("train")).reset_index(drop=True)
    test_df = add_engineered_features(load_merged("test")).reset_index(drop=True)
    if args.sample_size is not None and args.sample_size < len(train_df):
        train_df = train_df.sample(args.sample_size, random_state=42).sort_index().reset_index(drop=True)

    models = fit_full_models(train_df, structured_weight=args.structured_weight)
    save_models(models, ARTIFACTS_DIR / "models" / "ensemble.joblib")

    predictions = predict_with_full_models(models, test_df)
    submission = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            TARGET_COLUMN: predictions,
        }
    )
    submission.to_csv(ARTIFACTS_DIR / "submission.csv", index=False)
    print(f"saved {ARTIFACTS_DIR / 'submission.csv'}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
