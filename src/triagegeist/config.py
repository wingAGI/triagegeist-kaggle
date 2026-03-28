"""Project configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ARCHIVE = PROJECT_ROOT / "data" / "triagegeist.zip"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ID_COLUMN = "patient_id"
TARGET_COLUMN = "triage_acuity"
TEXT_COLUMN = "chief_complaint_raw"

LEAKAGE_COLUMNS = [
    ID_COLUMN,
    TARGET_COLUMN,
    "disposition",
    "ed_los_hours",
]

SUBGROUP_COLUMNS = [
    "age_group",
    "sex",
    "language",
    "site_id",
    "arrival_mode",
]

HIGH_RISK_THRESHOLD = 2

