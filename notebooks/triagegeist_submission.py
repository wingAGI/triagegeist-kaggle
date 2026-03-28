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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from triagegeist.config import ARTIFACTS_DIR, HIGH_RISK_THRESHOLD, ID_COLUMN, LEAKAGE_COLUMNS, TARGET_COLUMN, TEXT_COLUMN
from triagegeist.data import load_merged, load_raw_tables
from triagegeist.features import add_engineered_features

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 160)

SEED = 42


# %% [markdown]
# ## 2. Data And Leakage Check
#
# We join the competition tables on `patient_id` and explicitly exclude `disposition` and `ed_los_hours` from training.
# Those fields are useful for post-triage analysis but are not available at the time of the triage decision.

# %%
tables = load_raw_tables()
train = load_merged("train")
test = load_merged("test")

display(pd.DataFrame(
    {
        "table": list(tables.keys()),
        "rows": [len(df) for df in tables.values()],
        "columns": [df.shape[1] for df in tables.values()],
    }
))

display(
    pd.DataFrame(
        {
            "excluded_column": LEAKAGE_COLUMNS,
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
# We also add a few simple clinically motivated indicators such as low oxygen, fever, tachycardia, and keyword flags.

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
train[TARGET_COLUMN].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title("Target Distribution")
ax.set_xlabel("ESI acuity")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Model Design
#
# The final project uses a structured model for tabular features and a lightweight text model for complaint text.
# Their probabilities are blended into a simple ensemble. The exact training logic lives in the local project scripts
# and should be kept synchronized with the notebook.

# %%
# NOTE:
# The actual training run for this project is performed by the local scripts in `scripts/`.
# In the Kaggle notebook, replace this cell with the final training code or load precomputed artifacts
# if you are using the notebook primarily for presentation.

cv_summary_path = ARTIFACTS_DIR / "metrics.json"
subgroup_path = ARTIFACTS_DIR / "tables" / "subgroup_metrics.csv"
confusion_path = ARTIFACTS_DIR / "tables" / "confusion_matrix.csv"
undertriage_path = ARTIFACTS_DIR / "tables" / "undertriage_examples.csv"


# %% [markdown]
# ## 5. Main Results
#
# The main results should report macro-F1, high-risk recall, class-wise metrics, and a normalized confusion matrix.
# These values are read from the local artifacts generated during the development pipeline.

# %%
if cv_summary_path.exists():
    metrics = json.loads(cv_summary_path.read_text())
    display(pd.DataFrame([metrics]))
else:
    metrics = {}
    print("No local metrics found yet. Run the training pipeline first.")

if confusion_path.exists():
    confusion = pd.read_csv(confusion_path)
    display(confusion)
else:
    confusion = pd.DataFrame()


# %% [markdown]
# ## 6. Undertriage Analysis
#
# Severe undertriage is defined as cases where the model predicts a substantially less urgent level than the label,
# or where clinically important high-risk patients are missed. The table below should be replaced with the final
# set of false negatives and highest-risk misses from the artifacts.

# %%
if undertriage_path.exists():
    undertriage_examples = pd.read_csv(undertriage_path)
    display(undertriage_examples.head(15))
else:
    undertriage_examples = pd.DataFrame()
    print("No undertriage table found yet.")


# %% [markdown]
# ## 7. Subgroup Audit
#
# We want to understand whether performance is stable across `site_id`, `language`, `arrival_mode`, and `age_group`.
# The goal is not to claim perfect fairness, but to surface the slices that deserve caution.

# %%
if subgroup_path.exists():
    subgroup_metrics = pd.read_csv(subgroup_path)
    display(subgroup_metrics.sort_values(["subgroup", "macro_f1"], ascending=[True, False]).head(30))
else:
    subgroup_metrics = pd.DataFrame()
    print("No subgroup audit found yet.")


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
case_studies = undertriage_examples.head(5).copy() if not undertriage_examples.empty else pd.DataFrame()
display(case_studies)


# %% [markdown]
# ## 9. Limitations
#
# - The dataset is synthetic, so external validity is limited.
# - This notebook demonstrates a static retrospective workflow, not prospective validation.
# - The model is a second-reader decision-support tool and should not replace triage staff.
# - Real deployment would require calibration, monitoring, and clinical review.

# %%
limitations = [
    "Synthetic data limits external validity.",
    "No prospective validation in a live ED workflow.",
    "Model output is support only, not a replacement for clinical judgment.",
]
display(pd.DataFrame({"limitations": limitations}))


# %% [markdown]
# ## 10. Reproducibility
#
# Exact commands used during development:
# - `python scripts/run_pipeline.py --sample-size 12000 --folds 2 --structured-weight 0.8 --skip-final-fit`
# - `python scripts/fit_submission.py --structured-weight 0.8`
#
# Artifacts are written under `artifacts/`:
# - metrics
# - confusion matrix
# - subgroup audit
# - undertriage examples
# - model bundle
# - `submission.csv`

# %%
artifact_listing = pd.DataFrame(
    {
        "artifact": [
            "metrics.json",
            "tables/confusion_matrix.csv",
            "tables/subgroup_metrics.csv",
            "tables/undertriage_examples.csv",
            "figures/confusion_matrix.png",
            "figures/subgroup_macro_f1.png",
            "models/ensemble.joblib",
            "submission.csv",
        ],
        "exists": [
            (ARTIFACTS_DIR / "metrics.json").exists(),
            (ARTIFACTS_DIR / "tables" / "confusion_matrix.csv").exists(),
            (ARTIFACTS_DIR / "tables" / "subgroup_metrics.csv").exists(),
            (ARTIFACTS_DIR / "tables" / "undertriage_examples.csv").exists(),
            (ARTIFACTS_DIR / "figures" / "confusion_matrix.png").exists(),
            (ARTIFACTS_DIR / "figures" / "subgroup_macro_f1.png").exists(),
            (ARTIFACTS_DIR / "models" / "ensemble.joblib").exists(),
            (ARTIFACTS_DIR / "submission.csv").exists(),
        ],
    }
)
display(artifact_listing)


# %% [markdown]
# ## Conclusion
#
# Replace this placeholder with the final narrative:
# - what the model does well
# - where it still fails
# - why the remaining errors matter clinically
# - how the notebook supports safer triage rather than replacing clinicians
