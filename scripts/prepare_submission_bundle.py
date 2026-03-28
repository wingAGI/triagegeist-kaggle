#!/usr/bin/env python
"""Collect the local submission assets into a single folder."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "submission_bundle"

FILES = [
    ROOT / "artifacts" / "submission.csv",
    ROOT / "artifacts" / "metrics.json",
    ROOT / "artifacts" / "run_summary.json",
    ROOT / "artifacts" / "figures" / "confusion_matrix.png",
    ROOT / "artifacts" / "figures" / "subgroup_macro_f1.png",
    ROOT / "artifacts" / "tables" / "subgroup_metrics.csv",
    ROOT / "artifacts" / "tables" / "undertriage_examples.csv",
    ROOT / "notebooks" / "triagegeist_kaggle_ready.ipynb",
    ROOT / "docs" / "WRITEUP_DRAFT.md",
    ROOT / "docs" / "SUBMISSION_CHECKLIST.md",
    ROOT / "docs" / "SUBMISSION_STATUS.md",
    ROOT / "assets" / "cover_v1.png",
]


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True, exist_ok=True)

    for file_path in FILES:
        if not file_path.exists():
            continue
        target = BUNDLE / file_path.name
        shutil.copy2(file_path, target)

    print(f"prepared {BUNDLE}")
    for path in sorted(BUNDLE.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
