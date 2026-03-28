# Submission Checklist

Use this list before clicking `Submit` on Kaggle. The goal is to catch the boring failures that cost a good solution its chance.

## Notebook Runability

- The notebook runs end to end from a fresh session.
- All data loads succeed from the expected Kaggle paths.
- No cell depends on hidden local state, manual edits, or private files.
- The notebook completes within a reasonable runtime for Kaggle execution.
- The final notebook output includes the key results needed for the writeup.
- Any randomness is controlled with a fixed seed.
- The notebook does not use leakage columns such as `disposition` or `ed_los_hours` for training.

## Data And Leakage

- Train, test, chief complaint, and patient history tables are joined correctly.
- Missing values are handled explicitly and described in the notebook.
- Feature engineering is documented clearly enough for reproduction.
- Any derived feature is clinically motivated and does not leak the target.
- The analysis section states why excluded columns are not valid predictors at triage time.

## Model And Results

- The final model is clearly identified in the notebook.
- Cross-validation or holdout evaluation is reported.
- Macro-F1 and class-level behavior are shown.
- Confusion matrix is included.
- High-risk recall or undertriage behavior is reported.
- Any ablation or comparison table is present if used in the final argument.

## Writeup Requirements

- The writeup has a title and subtitle.
- The clinical problem statement is explicit.
- The dataset description includes the synthetic-data caveat.
- The method section explains the model and feature design.
- Leakage prevention is stated plainly.
- Results include the main metric, error structure, and subgroup behavior.
- Undertriage analysis is described with at least a few concrete examples.
- Limitations and deployment boundaries are stated without overclaiming.
- Reproducibility notes are included.
- The writeup stays within the competition word limit.

## Required Attachments

- A cover image is attached and matches the required dimensions.
- A public Kaggle notebook is attached to the writeup.
- A public project link is attached to the writeup.
- The project link is reachable without login or paywall.
- If the link points to a repository, it includes setup and run instructions.

## Final Submission Sanity Check

- The notebook is set to public.
- The writeup is set to submitted, not draft.
- The notebook attachment points to the correct file.
- The cover image is the intended final version.
- The project link is the intended final version.
- The final submission is attached before the deadline.
- The submitted version matches the notebook and writeup you actually reviewed.

## Quick Last Look

Before submitting, read these three statements out loud and make sure they are true:

- This is a clinical decision-support prototype, not a generic classification demo.
- The model improves safety by reducing severe undertriage risk, not by claiming perfect triage.
- The results are reproducible, caveated, and honest about synthetic-data limits.
