# ROGII Wellbore Geology Prediction Notes

## Confirmed From Local Files

- The task is to predict `TVT` for hidden horizontal-well intervals.
- The local task PDF identifies RMSE as the evaluation metric.
- Training horizontal-well files contain `TVT`.
- Test horizontal-well files contain `TVT_input`, with missing rows corresponding to required predictions.
- The submission file has exactly two columns: `id,tvt`.
- The local sample submission has 14,151 rows.

## Local Data Layout

- `data/train/*__horizontal_well.csv`: horizontal wells with target labels.
- `data/train/*__typewell.csv`: type-well context tables.
- `data/train/*.png`: image files provided with training wells.
- `data/test/*__horizontal_well.csv`: horizontal wells with hidden `TVT_input` intervals.
- `data/test/*__typewell.csv`: type-well context tables.
- `data/sample_submission.csv`: required submission order and schema.

## Current Baseline Scope

- Uses tabular trajectory and gamma-ray features only.
- Uses filename prefixes as well ids.
- Uses well-level `GroupKFold` validation.
- Excludes direct `TVT` and direct `TVT_input` features to avoid target leakage.
- Does not use PNG/image modeling.
- Does not use external data.

## Items Requiring Manual Confirmation

The Kaggle Rules page was not available as usable command-line text during setup. The following must be manually checked in a browser before any competition-sensitive changes:

- External-data policy.
- Team merger rules.
- Private sharing restrictions.
- Code sharing restrictions.
- Submission limits.
- Eligibility, prize, and licensing terms.
