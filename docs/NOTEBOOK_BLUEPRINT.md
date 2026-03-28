# Notebook Blueprint

## 1. Problem Framing

- Explain the clinical cost of undertriage.
- State that the system is a decision-support layer.

## 2. Data And Leakage Check

- Show all tables and joins.
- Explicitly note that `disposition` and `ed_los_hours` are excluded from training.

## 3. Feature Engineering

- Summarize structured features, history flags, and complaint text.
- Visualize class distribution and selected vital-sign gradients.

## 4. Model Design

- Structured model diagram
- Text model diagram
- Ensemble weighting rationale

## 5. Main Results

- CV table
- Confusion matrix
- Per-class metrics table

## 6. Undertriage Analysis

- Severe undertriage definition
- High-risk recall
- Top missed cases table

## 7. Subgroup Audit

- Macro-F1 by site, language, arrival mode, age group
- High-risk recall by subgroup

## 8. Case Studies

- 3 to 5 example records
- Each case should show true label, predicted label, risk context, and clinical interpretation

## 9. Limitations

- Synthetic data
- No prospective validation
- Cannot replace clinical judgment

## 10. Reproducibility

- Exact training command
- Output artifact locations
