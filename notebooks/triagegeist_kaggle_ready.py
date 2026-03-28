# %% [markdown]
# # Beyond Acuity Prediction: An Interpretable Triage Support Pipeline for Undertriage Detection
#
# Structured vitals, complaint text, and patient history for emergency triage decision support and subgroup auditing.

# %% [markdown]
# ## 1. Problem Framing
#
# Emergency triage is a high-stakes decision point. The goal of this notebook is to build a second-reader safety layer
# that supports triage staff by predicting acuity, flagging likely undertriage, and surfacing subgroup-specific failure
# modes. The model is intentionally framed as decision support, not replacement.

# %%
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 200)

SEED = 42
N_FOLDS = 3
STRUCTURED_WEIGHT = 0.8
HIGH_RISK_THRESHOLD = 2

TARGET = "triage_acuity"
ID_COL = "patient_id"
TEXT_COL = "chief_complaint_raw"
LEAKAGE_COLS = ["disposition", "ed_los_hours"]


def _candidate_data_roots() -> list[Path]:
    roots = [
        Path("/kaggle/input/triagegeist"),
        Path("/kaggle/input/triagegeist-competition"),
        Path("/kaggle/input"),
        Path("data"),
    ]
    return roots


def _resolve_data_source() -> tuple[str, Path]:
    for root in _candidate_data_roots():
        if root.is_dir() and (root / "train.csv").exists():
            return "dir", root
        if root.is_file() and root.suffix == ".zip":
            return "zip", root

    local_zip = Path("data/triagegeist.zip")
    if local_zip.exists():
        return "zip", local_zip

    raise FileNotFoundError(
        "Could not find the competition data. Expected /kaggle/input/triagegeist or data/triagegeist.zip."
    )


DATA_KIND, DATA_SOURCE = _resolve_data_source()


def load_csv(name: str) -> pd.DataFrame:
    if DATA_KIND == "dir":
        return pd.read_csv(DATA_SOURCE / name)
    with zipfile.ZipFile(DATA_SOURCE) as archive:
        with archive.open(name) as handle:
            return pd.read_csv(handle)


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "train": load_csv("train.csv"),
        "test": load_csv("test.csv"),
        "chief_complaints": load_csv("chief_complaints.csv"),
        "patient_history": load_csv("patient_history.csv"),
        "sample_submission": load_csv("sample_submission.csv"),
    }


def merge_tables(split: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables[split].merge(
        tables["chief_complaints"][[ID_COL, TEXT_COL]],
        on=ID_COL,
        how="left",
    ).merge(
        tables["patient_history"],
        on=ID_COL,
        how="left",
    )
    return frame


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    complaint = df[TEXT_COL].fillna("").astype(str).str.lower()

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

    keyword_patterns = {
        "kw_chest_pain": r"chest pain|thoracic pain|crushing chest",
        "kw_stroke_neuro": r"stroke|seizure|thunderclap|loss of vision|weakness|aphasia",
        "kw_respiratory_distress": r"shortness of breath|asthma|hypoxia|wheeze|near-drowning",
        "kw_trauma": r"trauma|fracture|haemothorax|stab|wound|fall|injury",
        "kw_overdose_toxic": r"overdose|poison|toxic|substance",
        "kw_bleeding": r"bleed|haemorrhage|hemorrhage|melena|hematemesis",
        "kw_pregnancy": r"pregnan|ectopic|postpartum|miscarriage",
        "kw_infection_sepsis": r"sepsis|fever|necrotising|infection|cellulitis",
    }
    for feature_name, pattern in keyword_patterns.items():
        df[feature_name] = complaint.str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)

    return df


def build_feature_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {ID_COL, TARGET, TEXT_COL, *LEAKAGE_COLS}
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def build_structured_model(frame: pd.DataFrame) -> Pipeline:
    numeric_columns, categorical_columns = build_feature_groups(frame)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
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
        max_depth=7,
        max_iter=220,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=SEED,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


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
                    max_features=25000,
                    sublinear_tf=True,
                ),
            ),
            ("model", ComplementNB(alpha=0.25)),
        ]
    )


def macro_f1(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def high_risk_recall(y_true: pd.Series, y_pred: np.ndarray) -> float:
    mask = y_true <= HIGH_RISK_THRESHOLD
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[mask.to_numpy()] <= HIGH_RISK_THRESHOLD))


def undertriage_rate(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true.to_numpy()) >= 2))


def compute_classification_table(y_true: pd.Series, y_pred: np.ndarray, label: str) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1, 2, 3, 4, 5],
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "model": label,
            "class": [1, 2, 3, 4, 5],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def compute_subgroup_metrics(metadata: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subgroup in ["age_group", "sex", "language", "site_id", "arrival_mode"]:
        for value, group in metadata.groupby(subgroup, dropna=False):
            if len(group) < 100:
                continue
            idx = group.index.to_numpy()
            rows.append(
                {
                    "subgroup": subgroup,
                    "value": value,
                    "count": int(len(idx)),
                    "macro_f1": macro_f1(y_true.loc[idx], y_pred[idx]),
                    "high_risk_recall": high_risk_recall(y_true.loc[idx], y_pred[idx]),
                    "undertriage_rate": undertriage_rate(y_true.loc[idx], y_pred[idx]),
                }
            )
    return pd.DataFrame(rows).sort_values(["subgroup", "macro_f1", "count"], ascending=[True, False, False])


def plot_confusion_matrix(confusion_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(confusion_df.values, cmap="Blues")
    ax.set_xticks(range(len(confusion_df.columns)))
    ax.set_xticklabels(confusion_df.columns)
    ax.set_yticks(range(len(confusion_df.index)))
    ax.set_yticklabels(confusion_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Normalized Confusion Matrix")
    for row_idx in range(confusion_df.shape[0]):
        for col_idx in range(confusion_df.shape[1]):
            ax.text(col_idx, row_idx, f"{confusion_df.iloc[row_idx, col_idx]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_subgroup_bars(subgroup_df: pd.DataFrame) -> None:
    if subgroup_df.empty:
        return
    top = subgroup_df.sort_values("macro_f1").groupby("subgroup").head(5)
    labels = [f"{row.subgroup}: {row.value}" for row in top.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, top["macro_f1"])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Subgroup Performance Snapshot")
    plt.tight_layout()
    plt.show()


def fit_cv_ensemble(train_df: pd.DataFrame, test_df: pd.DataFrame, folds: int = N_FOLDS, structured_weight: float = STRUCTURED_WEIGHT):
    y = train_df[TARGET].copy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    structured_model = build_structured_model(train_df)
    text_model = build_text_model()

    classes = np.array(sorted(y.unique()))
    oof_structured = np.zeros((len(train_df), len(classes)))
    oof_text = np.zeros((len(train_df), len(classes)))
    test_structured = np.zeros((len(test_df), len(classes)))
    test_text = np.zeros((len(test_df), len(classes)))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df, y), start=1):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)

        structured_fold = clone(structured_model)
        structured_fold.fit(fold_train, fold_train[TARGET])
        oof_structured[valid_idx] = structured_fold.predict_proba(fold_valid)
        test_structured += structured_fold.predict_proba(test_df) / folds

        text_fold = clone(text_model)
        text_fold.fit(fold_train[TEXT_COL], fold_train[TARGET])
        oof_text[valid_idx] = text_fold.predict_proba(fold_valid[TEXT_COL])
        test_text += text_fold.predict_proba(test_df[TEXT_COL]) / folds

        print(f"completed fold {fold}/{folds}")

    oof_ensemble = structured_weight * oof_structured + (1 - structured_weight) * oof_text
    test_ensemble = structured_weight * test_structured + (1 - structured_weight) * test_text

    predictions = {
        "structured": classes[np.argmax(oof_structured, axis=1)],
        "text": classes[np.argmax(oof_text, axis=1)],
        "ensemble": classes[np.argmax(oof_ensemble, axis=1)],
    }
    probs = {"structured": oof_structured, "text": oof_text, "ensemble": oof_ensemble}

    metrics_rows = []
    class_rows = []
    for name in ["structured", "text", "ensemble"]:
        pred = predictions[name]
        metrics_rows.append(
            {
                "model": name,
                "macro_f1": macro_f1(y, pred),
                "high_risk_recall": high_risk_recall(y, pred),
                "undertriage_rate": undertriage_rate(y, pred),
            }
        )
        class_rows.append(compute_classification_table(y, pred, name))

    metrics_df = pd.DataFrame(metrics_rows)
    class_metrics_df = pd.concat(class_rows, ignore_index=True)

    confusion = pd.crosstab(
        pd.Series(y, name="actual"),
        pd.Series(predictions["ensemble"], name="predicted"),
        normalize="index",
    ).round(4)

    oof = train_df[[ID_COL, TEXT_COL, "age_group", "sex", "language", "site_id", "arrival_mode"]].copy()
    oof["actual_acuity"] = y.to_numpy()
    oof["predicted_acuity"] = predictions["ensemble"]
    oof["prediction_gap"] = oof["predicted_acuity"] - oof["actual_acuity"]
    oof["ensemble_high_risk_prob"] = oof_ensemble[:, :HIGH_RISK_THRESHOLD].sum(axis=1)
    oof["news2_score"] = train_df["news2_score"].to_numpy()
    oof["spo2"] = train_df["spo2"].to_numpy()
    oof["gcs_total"] = train_df["gcs_total"].to_numpy()
    oof = oof.sort_values(["prediction_gap", "ensemble_high_risk_prob", "news2_score"], ascending=[False, False, False])
    undertriage = oof[oof["prediction_gap"] >= 2].head(20).copy()
    subgroup = compute_subgroup_metrics(train_df, y, predictions["ensemble"])

    return {
        "metrics": metrics_df,
        "class_metrics": class_metrics_df,
        "confusion": confusion,
        "undertriage": undertriage,
        "subgroup": subgroup,
        "oof_predictions": predictions,
        "oof_probs": probs,
        "test_probs": test_ensemble,
        "classes": classes,
    }


def fit_final_models(train_df: pd.DataFrame, structured_weight: float = STRUCTURED_WEIGHT):
    structured_model = build_structured_model(train_df)
    text_model = build_text_model()
    structured_model.fit(train_df, train_df[TARGET])
    text_model.fit(train_df[TEXT_COL], train_df[TARGET])
    return {
        "structured": structured_model,
        "text": text_model,
        "structured_weight": structured_weight,
    }


def predict_submission(models: dict[str, object], test_df: pd.DataFrame) -> np.ndarray:
    structured_prob = models["structured"].predict_proba(test_df)
    text_prob = models["text"].predict_proba(test_df[TEXT_COL])
    blended = models["structured_weight"] * structured_prob + (1 - models["structured_weight"]) * text_prob
    classes = models["structured"].named_steps["model"].classes_
    return classes[np.argmax(blended, axis=1)]


# %% [markdown]
# ## 2. Data And Leakage Check
#
# We join the competition tables on `patient_id` and explicitly exclude `disposition` and `ed_los_hours` from training.
# Those fields are useful for post-triage analysis but are not available at the time of the triage decision.

# %%
tables = load_tables()
train = merge_tables("train", tables)
test = merge_tables("test", tables)
sample_submission = tables["sample_submission"].copy()

display(
    pd.DataFrame(
        {
            "table": list(tables.keys()),
            "rows": [len(df) for df in tables.values()],
            "columns": [df.shape[1] for df in tables.values()],
        }
    )
)

display(
    pd.DataFrame(
        {
            "excluded_column": ["patient_id", TARGET, "disposition", "ed_los_hours"],
            "why_excluded": [
                "identifier, not a feature",
                "target variable",
                "post-triage outcome",
                "post-triage outcome / length of stay",
            ],
        }
    )
)

missing_train = train.isna().mean().sort_values(ascending=False).head(12).to_frame("missing_rate")
missing_test = test.isna().mean().sort_values(ascending=False).head(12).to_frame("missing_rate")
display(missing_train)
display(missing_test)

# %% [markdown]
# ## 3. Feature Engineering
#
# This project uses three feature families:
# - structured vitals and triage context
# - patient history flags
# - chief complaint text
#
# We also add a few clinically motivated indicators such as low oxygen, fever, tachycardia, and complaint keyword flags.

# %%
train_feat = add_engineered_features(train)
test_feat = add_engineered_features(test)

feature_preview_columns = [
    "news2_score",
    "flag_low_oxygen",
    "flag_fever",
    "flag_tachycardia",
    "flag_tachypnea",
    "flag_hypotension",
    "cardio_burden",
    "respiratory_burden",
    "neuro_burden",
    "frailty_burden",
    "kw_chest_pain",
    "kw_stroke_neuro",
    "kw_respiratory_distress",
]

display(train_feat[feature_preview_columns].head())

fig, ax = plt.subplots(figsize=(8, 4))
train[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title("Target Distribution")
ax.set_xlabel("ESI acuity")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Model Design
#
# The final project uses a structured model for tabular features and a lightweight text model for complaint text.
# Their probabilities are blended into a simple ensemble.

# %%
results = fit_cv_ensemble(train_feat, test_feat, folds=N_FOLDS, structured_weight=STRUCTURED_WEIGHT)
display(results["metrics"])
display(results["class_metrics"])

# %% [markdown]
# ## 5. Main Results
#
# The main results report macro-F1, high-risk recall, class-wise metrics, and a normalized confusion matrix.

# %%
display(results["confusion"])
plot_confusion_matrix(results["confusion"])

# %% [markdown]
# ## 6. Undertriage Analysis
#
# Severe undertriage is defined as cases where the model predicts a substantially less urgent level than the label,
# or where clinically important high-risk patients are missed.

# %%
display(results["undertriage"])

# %% [markdown]
# ## 7. Subgroup Audit
#
# We want to understand whether performance is stable across `site_id`, `language`, `arrival_mode`, and `age_group`.
# The goal is not to claim perfect fairness, but to surface the slices that deserve caution.

# %%
display(results["subgroup"].sort_values(["subgroup", "macro_f1"], ascending=[True, False]).head(30))
plot_subgroup_bars(results["subgroup"])

# %% [markdown]
# ## 8. Case Studies
#
# Add 3 to 5 short case studies here. Each should show:
# - true acuity
# - predicted acuity
# - complaint text
# - vitals / risk context
# - a short clinical interpretation

# %%
case_studies = results["undertriage"].head(5).copy()
display(case_studies)

# %% [markdown]
# ## 9. Limitations
#
# - The dataset is synthetic, so external validity is limited.
# - This notebook demonstrates a static retrospective workflow, not prospective validation.
# - The model is a second-reader decision-support tool and should not replace triage staff.
# - Real deployment would require calibration, monitoring, and clinical review.

# %%
limitations = pd.DataFrame(
    {
        "limitation": [
            "Synthetic data limits external validity.",
            "No prospective validation in a live ED workflow.",
            "Model output is support only, not a replacement for clinical judgment.",
        ]
    }
)
display(limitations)

# %% [markdown]
# ## 10. Reproducibility
#
# The notebook runs end-to-end and writes `submission.csv` in the current working directory.
# Artifacts shown above are generated directly in the notebook, so the report is fully portable.

# %%
final_models = fit_final_models(train_feat, structured_weight=STRUCTURED_WEIGHT)
submission_preds = predict_submission(final_models, test_feat)
submission = sample_submission.copy()
submission[TARGET] = submission_preds
submission.to_csv("submission.csv", index=False)

display(submission.head())
display(
    pd.DataFrame(
        {
            "output": [
                "submission.csv",
                "cv metrics table",
                "confusion matrix",
                "subgroup audit",
                "undertriage examples",
            ],
            "status": ["written", "displayed", "displayed", "displayed", "displayed"],
        }
    )
)

print(json.dumps(results["metrics"].to_dict(orient="records"), indent=2))
