# Kaggle Writeup Outline

## Title

Beyond Acuity Prediction: An Interpretable Triage Support Pipeline for Undertriage Detection

## Subtitle

Structured vitals, complaint text, and patient history for emergency triage decision support and subgroup auditing

## Sections

1. Clinical problem and why undertriage matters
2. Dataset description and synthetic-data caveat
3. Task framing
4. Leakage prevention choices
5. Modeling approach
6. Main performance results
7. Undertriage and high-risk recall analysis
8. Subgroup audit
9. Case studies
10. Limitations and deployment boundaries
11. Reproducibility notes

## Must-Hit Claims

- The model is a second-reader safety layer, not a replacement for triage staff.
- Avoiding severe undertriage is more clinically meaningful than maximizing a single aggregate score.
- Performance is reported alongside subgroup behavior and error structure.
- Synthetic data limits external validity and requires prospective validation.
