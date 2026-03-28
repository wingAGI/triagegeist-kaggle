"""Model construction and training logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import HIGH_RISK_THRESHOLD, ID_COLUMN, LEAKAGE_COLUMNS, TARGET_COLUMN, TEXT_COLUMN
from .features import feature_column_groups


@dataclass
class TrainingArtifacts:
    oof_probabilities: np.ndarray
    test_probabilities: np.ndarray
    predictions: np.ndarray
    metrics: dict[str, float]
    subgroup_metrics: pd.DataFrame
    confusion_table: pd.DataFrame
    undertriage_examples: pd.DataFrame
    fitted_models: list[dict[str, object]]


def _drop_leakage_columns(frame: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [column for column in LEAKAGE_COLUMNS if column in frame.columns]
    return frame.drop(columns=drop_columns).copy()


def build_structured_model(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_text_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            ("model", ComplementNB(alpha=0.3)),
        ]
    )


def _macro_f1(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def _high_risk_recall(y_true: pd.Series, y_pred: np.ndarray) -> float:
    mask = y_true <= HIGH_RISK_THRESHOLD
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[mask.to_numpy()] <= HIGH_RISK_THRESHOLD))


def _undertriage_rate(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true.to_numpy()) >= 2))


def compute_subgroup_metrics(
    metadata: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    subgroup_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in subgroup_columns:
        for value, group in metadata.groupby(column, dropna=False):
            if len(group) < 100:
                continue
            rows.append(
                {
                    "subgroup": column,
                    "value": value,
                    "count": int(len(group)),
                    "macro_f1": _macro_f1(y_true.loc[group.index], y_pred[group.index.to_numpy()]),
                    "high_risk_recall": _high_risk_recall(
                        y_true.loc[group.index], y_pred[group.index.to_numpy()]
                    ),
                    "undertriage_rate": _undertriage_rate(
                        y_true.loc[group.index], y_pred[group.index.to_numpy()]
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["subgroup", "value", "count", "macro_f1", "high_risk_recall", "undertriage_rate"]
        )
    return pd.DataFrame(rows).sort_values(
        ["subgroup", "macro_f1", "count"], ascending=[True, False, False]
    )


def build_confusion_table(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    table = pd.crosstab(
        pd.Series(y_true, name="actual"),
        pd.Series(y_pred, name="predicted"),
        normalize="index",
    )
    return table.round(4)


def undertriage_examples(
    source_frame: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    columns = [
        ID_COLUMN,
        TEXT_COLUMN,
        "news2_score",
        "spo2",
        "gcs_total",
        "age_group",
        "language",
        "arrival_mode",
    ]
    report = source_frame[columns].copy()
    report["actual_acuity"] = y_true.to_numpy()
    report["predicted_acuity"] = y_pred
    report["prediction_gap"] = report["predicted_acuity"] - report["actual_acuity"]
    report["high_risk_probability"] = probabilities[:, :HIGH_RISK_THRESHOLD].sum(axis=1)
    report = report.sort_values(
        ["prediction_gap", "high_risk_probability", "news2_score"],
        ascending=[False, False, False],
    )
    return report[report["prediction_gap"] >= 2].head(25)


def train_ensemble(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    subgroup_columns: list[str],
    sample_size: int | None = None,
    folds: int = 3,
    structured_weight: float = 0.8,
) -> TrainingArtifacts:
    if sample_size is not None and sample_size < len(train_frame):
        train_frame = train_frame.sample(sample_size, random_state=42).sort_index()
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    y_train = train_frame[TARGET_COLUMN].copy()
    X_train = _drop_leakage_columns(train_frame)
    X_test = _drop_leakage_columns(test_frame)

    numeric_columns, categorical_columns, _ = feature_column_groups(X_train)
    structured_model = build_structured_model(numeric_columns, categorical_columns)
    text_model = build_text_model()

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    classes = np.sort(y_train.unique())
    oof_structured = np.zeros((len(X_train), len(classes)))
    oof_text = np.zeros((len(X_train), len(classes)))
    test_structured = np.zeros((len(X_test), len(classes)))
    test_text = np.zeros((len(X_test), len(classes)))
    fitted_models: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
        train_fold_X = X_train.iloc[train_idx]
        train_fold_y = y_train.iloc[train_idx]
        valid_fold_X = X_train.iloc[valid_idx]

        structured_fold = clone(structured_model)
        structured_fold.fit(train_fold_X, train_fold_y)
        oof_structured[valid_idx] = structured_fold.predict_proba(valid_fold_X)
        test_structured += structured_fold.predict_proba(X_test) / folds

        text_fold = clone(text_model)
        text_fold.fit(train_fold_X[TEXT_COLUMN], train_fold_y)
        oof_text[valid_idx] = text_fold.predict_proba(valid_fold_X[TEXT_COLUMN])
        test_text += text_fold.predict_proba(X_test[TEXT_COLUMN]) / folds

        fitted_models.append({"fold": fold, "structured": structured_fold, "text": text_fold})

    oof_probabilities = structured_weight * oof_structured + (1.0 - structured_weight) * oof_text
    test_probabilities = structured_weight * test_structured + (1.0 - structured_weight) * test_text
    predictions = classes[np.argmax(oof_probabilities, axis=1)]

    metrics = {
        "macro_f1": _macro_f1(y_train, predictions),
        "high_risk_recall": _high_risk_recall(y_train, predictions),
        "undertriage_rate": _undertriage_rate(y_train, predictions),
    }
    subgroup_metrics = compute_subgroup_metrics(X_train, y_train, predictions, subgroup_columns)
    confusion_table = build_confusion_table(y_train, predictions)
    examples = undertriage_examples(train_frame, y_train, predictions, oof_probabilities)

    return TrainingArtifacts(
        oof_probabilities=oof_probabilities,
        test_probabilities=test_probabilities,
        predictions=predictions,
        metrics=metrics,
        subgroup_metrics=subgroup_metrics,
        confusion_table=confusion_table,
        undertriage_examples=examples,
        fitted_models=fitted_models,
    )


def fit_full_models(train_frame: pd.DataFrame, structured_weight: float = 0.8) -> dict[str, object]:
    train_frame = train_frame.reset_index(drop=True)
    X_train = _drop_leakage_columns(train_frame)
    y_train = train_frame[TARGET_COLUMN].copy()
    numeric_columns, categorical_columns, _ = feature_column_groups(X_train)

    structured_model = build_structured_model(numeric_columns, categorical_columns)
    text_model = build_text_model()
    structured_model.fit(X_train, y_train)
    text_model.fit(X_train[TEXT_COLUMN], y_train)
    return {
        "structured": structured_model,
        "text": text_model,
        "structured_weight": structured_weight,
    }


def predict_with_full_models(models: dict[str, object], test_frame: pd.DataFrame) -> np.ndarray:
    X_test = _drop_leakage_columns(test_frame.reset_index(drop=True))
    structured_prob = models["structured"].predict_proba(X_test)
    text_prob = models["text"].predict_proba(X_test[TEXT_COLUMN])
    structured_weight = models["structured_weight"]
    ensemble = structured_weight * structured_prob + (1.0 - structured_weight) * text_prob
    classes = models["structured"].named_steps["model"].classes_
    return classes[np.argmax(ensemble, axis=1)]


def save_models(models: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output_path)
