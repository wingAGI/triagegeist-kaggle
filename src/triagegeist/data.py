"""Dataset loading helpers."""

from __future__ import annotations

from zipfile import ZipFile

import pandas as pd

from .config import DATA_ARCHIVE, ID_COLUMN, TEXT_COLUMN


def _read_csv_from_archive(filename: str) -> pd.DataFrame:
    with ZipFile(DATA_ARCHIVE) as archive:
        with archive.open(filename) as handle:
            return pd.read_csv(handle)


def load_raw_tables() -> dict[str, pd.DataFrame]:
    return {
        "train": _read_csv_from_archive("train.csv"),
        "test": _read_csv_from_archive("test.csv"),
        "chief_complaints": _read_csv_from_archive("chief_complaints.csv"),
        "patient_history": _read_csv_from_archive("patient_history.csv"),
        "sample_submission": _read_csv_from_archive("sample_submission.csv"),
    }


def load_merged(split: str) -> pd.DataFrame:
    tables = load_raw_tables()
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    merged = tables[split].merge(
        tables["chief_complaints"][[ID_COLUMN, TEXT_COLUMN]],
        on=ID_COLUMN,
        how="left",
    ).merge(
        tables["patient_history"],
        on=ID_COLUMN,
        how="left",
    )
    return merged
