"""Feature engineering utilities."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import ID_COLUMN, LEAKAGE_COLUMNS, TARGET_COLUMN, TEXT_COLUMN

KEYWORD_PATTERNS = {
    "kw_chest_pain": r"chest pain|thoracic pain|crushing chest",
    "kw_stroke_neuro": r"stroke|seizure|thunderclap|loss of vision|weakness|aphasia",
    "kw_respiratory_distress": r"shortness of breath|asthma|hypoxia|wheeze|near-drowning",
    "kw_trauma": r"trauma|fracture|haemothorax|stab|wound|fall|injury",
    "kw_overdose_toxic": r"overdose|poison|toxic|substance",
    "kw_bleeding": r"bleed|haemorrhage|hemorrhage|melena|hematemesis",
    "kw_pregnancy": r"pregnan|ectopic|postpartum|miscarriage",
    "kw_infection_sepsis": r"sepsis|fever|necrotising|infection|cellulitis",
}


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    complaint = df[TEXT_COLUMN].fillna("").str.lower()

    df["pain_unrecorded"] = (df["pain_score"] == -1).astype(int)
    df["pain_score_clean"] = df["pain_score"].replace(-1, np.nan)

    df["flag_low_oxygen"] = (df["spo2"] < 92).astype(int)
    df["flag_fever"] = (df["temperature_c"] >= 38.0).astype(int)
    df["flag_tachycardia"] = (df["heart_rate"] >= 100).astype(int)
    df["flag_tachypnea"] = (df["respiratory_rate"] >= 22).astype(int)
    df["flag_hypotension"] = (df["systolic_bp"] < 90).astype(int)
    df["flag_gcs_abnormal"] = (df["gcs_total"] < 15).astype(int)
    df["flag_high_news2"] = (df["news2_score"] >= 5).astype(int)
    df["flag_high_shock_index"] = (df["shock_index"] >= 0.9).astype(int)

    df["chief_complaint_len"] = complaint.str.len()
    df["chief_complaint_word_count"] = complaint.str.split().str.len()
    df["chief_complaint_has_comma"] = complaint.str.contains(",", regex=False).astype(int)

    cardio_cols = [
        "hx_hypertension",
        "hx_heart_failure",
        "hx_atrial_fibrillation",
        "hx_coronary_artery_disease",
        "hx_peripheral_vascular_disease",
        "hx_stroke_prior",
    ]
    respiratory_cols = ["hx_asthma", "hx_copd"]
    neuro_cols = ["hx_dementia", "hx_epilepsy", "hx_stroke_prior"]
    frailty_cols = ["hx_dementia", "hx_ckd", "hx_malignancy", "hx_immunosuppressed"]

    df["cardio_burden"] = df[cardio_cols].sum(axis=1)
    df["respiratory_burden"] = df[respiratory_cols].sum(axis=1)
    df["neuro_burden"] = df[neuro_cols].sum(axis=1)
    df["frailty_burden"] = df[frailty_cols].sum(axis=1)

    for feature_name, pattern in KEYWORD_PATTERNS.items():
        df[feature_name] = complaint.str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)

    return df


def split_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    df = add_engineered_features(frame)
    target = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    feature_columns = [column for column in df.columns if column not in LEAKAGE_COLUMNS]
    return df[feature_columns], target


def feature_column_groups(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    text_columns = [TEXT_COLUMN]
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in frame.columns:
        if column in text_columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns, text_columns
