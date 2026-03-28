# Current Results

## Latest Cross-Validation Snapshot

Source: `artifacts/metrics.json`

- Macro-F1: `0.8892`
- High-risk recall (`acuity <= 2`): `0.9713`
- Severe undertriage rate (`pred - true >= 2`): `0.0016`

## What This Means

- The structured model is already strong enough to anchor a serious submission.
- The current story is strongest when framed around safety support, not raw classification alone.
- The remaining edge will come from writeup quality, undertriage case analysis, and subgroup audit clarity.

## Current Candidate Submission

- Full-train candidate exported to `artifacts/submission.csv`
- Model bundle saved to `artifacts/models/ensemble.joblib`
- Current structured/text blend weight: `0.8`

Current submission label distribution:

- Acuity `1`: `749`
- Acuity `2`: `3386`
- Acuity `3`: `7222`
- Acuity `4`: `5804`
- Acuity `5`: `2839`

## Strong Subgroup Snapshots

- `language = Estonian`: Macro-F1 `0.9129`
- `arrival_mode = walk-in`: Macro-F1 `0.8929`
- `site_id = SITE-TMP-01`: Macro-F1 `0.8946`

## Watch List

- `site_id = SITE-OUL-01`: Macro-F1 `0.8827`
- `language = Russian`: Macro-F1 `0.8771`
- `arrival_mode = transfer`: Macro-F1 `0.8795`

## Candidate Undertriage Cases

Good writeup examples from `artifacts/tables/undertriage_examples.csv`:

- `severe malaria, constant`
- `pleuritic chest pain, onset today`
- `palpitations with near-syncope`
