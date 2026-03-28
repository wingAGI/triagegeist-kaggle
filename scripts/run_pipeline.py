#!/usr/bin/env python
"""Run the local Triagegeist training and reporting pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from triagegeist.config import ARTIFACTS_DIR, ID_COLUMN, SUBGROUP_COLUMNS, TARGET_COLUMN
from triagegeist.data import load_merged
from triagegeist.features import add_engineered_features
from triagegeist.modeling import fit_full_models, predict_with_full_models, save_models, train_ensemble
from triagegeist.reporting import ensure_dir, plot_confusion_heatmap, plot_subgroup_bars, write_metrics, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--structured-weight", type=float, default=0.8)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-final-fit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(ARTIFACTS_DIR / "tables")
    ensure_dir(ARTIFACTS_DIR / "figures")
    ensure_dir(ARTIFACTS_DIR / "models")

    print("Loading merged data...")
    train_df = add_engineered_features(load_merged("train")).reset_index(drop=True)
    test_df = add_engineered_features(load_merged("test")).reset_index(drop=True)

    print("Running cross-validated ensemble...")
    training_artifacts = train_ensemble(
        train_frame=train_df,
        test_frame=test_df,
        subgroup_columns=SUBGROUP_COLUMNS,
        sample_size=args.sample_size,
        folds=args.folds,
        structured_weight=args.structured_weight,
    )

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    write_metrics(training_artifacts.metrics, metrics_path)
    write_table(
        training_artifacts.subgroup_metrics,
        ARTIFACTS_DIR / "tables" / "subgroup_metrics.csv",
    )
    training_artifacts.confusion_table.reset_index().to_csv(
        ARTIFACTS_DIR / "tables" / "confusion_matrix.csv", index=False
    )
    write_table(
        training_artifacts.undertriage_examples,
        ARTIFACTS_DIR / "tables" / "undertriage_examples.csv",
    )

    if not args.skip_figures:
        plot_confusion_heatmap(
            training_artifacts.confusion_table,
            ARTIFACTS_DIR / "figures" / "confusion_matrix.png",
        )
        plot_subgroup_bars(
            training_artifacts.subgroup_metrics,
            ARTIFACTS_DIR / "figures" / "subgroup_macro_f1.png",
        )

    summary = {
        "metrics": training_artifacts.metrics,
    }
    if not args.skip_final_fit:
        print("Fitting full models on all training data...")
        full_models = fit_full_models(train_df, structured_weight=args.structured_weight)
        save_models(full_models, ARTIFACTS_DIR / "models" / "ensemble.joblib")
        test_predictions = predict_with_full_models(full_models, test_df)

        submission = pd.DataFrame(
            {
                ID_COLUMN: test_df[ID_COLUMN],
                TARGET_COLUMN: test_predictions,
            }
        )
        submission.to_csv(ARTIFACTS_DIR / "submission.csv", index=False)
        summary["submission_path"] = str(ARTIFACTS_DIR / "submission.csv")
        summary["model_path"] = str(ARTIFACTS_DIR / "models" / "ensemble.joblib")
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
