# CLAUDE.md

This file provides repo-specific guidance for Claude Code when working in this project.

## Project Summary

This repository contains Kaggle ROGII Wellbore Geology Prediction experiments and submission notebooks. The task is to predict hidden `TVT` values for horizontal-well intervals and produce Kaggle-ready `submission.csv` files from notebooks.

## Primary Modeling Line

The primary modeling line in this repository is the **residual-anchor GBDT framework**:

- Train on `TVT - last_known_TVT_input`
- Reconstruct final `tvt` from the last known anchor plus predicted residual
- Prefer compact, leakage-aware, Kaggle-rerunnable notebooks

Current notebook progression:
- `kaggle_kernel_v1_lightgbm.ipynb` — stable baseline submission (Public LB 27.247)
- `kaggle_kernel_v3_residual_lgbm.ipynb` — first strong residual-anchor notebook (LB 14.527)
- `kaggle_kernel_v4_residual_optimized.ipynb` — residual optimization line (LB 13.764)
- `kaggle_kernel_v5_lgb_xgb_residual.ipynb` — LGB+XGB residual ensemble (LB 10.013)
- `kaggle_kernel_v6_regularized_robust_ensemble.ipynb` — anti-overfitting line (LB 9.791)
- `kaggle_kernel_v7_full_retrain.ipynb` — three leakage fixes + DTW features + full retrain
- `kaggle_kernel_v8_robust.ipynb` — log-driven robustness fixes over v7 (current active line)

## Kaggle Code Requirements (must-follow)

Every submitted notebook in this repo is bound by the competition's Code Requirements:

- **CPU/GPU run-time ≤ 9 hours**
- **No internet access** at submission run-time
- External data only allowed if attached as a Kaggle dataset (never network-fetched)
- **Submission filename must be `submission.csv`** at `/kaggle/working/submission.csv`

When working on any notebook in this repo:

- Never introduce `pip install`, `apt-get`, `wget`, `curl`, `urllib`, `requests`, or any network call. The legacy `subprocess pip install` block in v5–v8 is a try/except no-op safety net; it must remain `check=False` and the notebook must succeed when the install branch is skipped.
- Keep notebooks under 8 hours of estimated wall-clock to leave a safety margin. v7 ran ~7h; v8 estimated ~7.2–7.4h.
- The last cell of each notebook should assert the submission file name (`assert OUT.name == "submission.csv"`).
- If you add a heavy feature/model, document its incremental runtime cost and provide a degraded fallback (smaller fold count, fewer DTW radii, fewer seeds) that still emits `submission.csv`.

## V6-Specific Guidance

V6 should be treated as the current **robust experimental line**. Its purpose is not to exploit the public leaderboard but to improve hidden-leaderboard robustness.

### V6 design goals

- Keep the v5 residual-anchor structure
- Improve generalization, not just public-LB fit
- Avoid hidden-LB collapse
- Stay Kaggle memory-safe and rerunnable
- Preserve useful geology/spatial signal while removing brittle tricks

### V6 approved components

- Shuffled GroupKFold at the well level
- LightGBM with stronger regularization
- XGBoost for ensemble diversity
- CatBoost for ensemble diversity
- Absolute coordinates (`X`, `Y`, `Z`) retained for macro geology signal
- Relative trajectory features (`dx`, `dy`, `dz`, `dxy`)
- Clean `alpha * tau` post-processing only

### V6 forbidden / discouraged components

Do **not** add these into the primary V6 line unless the user explicitly asks and the rationale is documented:

- Exact `(X, Y, Z)` hard overlap replacement / direct label copying
- Public-LB-driven hill-climb weight search
- Savitzky-Golay smoothing on final predictions
- `w_pf` post-processing blend search
- TabICL dependencies or external artifact requirements
- Pseudo-labeling or public-test memorization tricks
- Large feature explosions without memory proof

## Validation Rules

### Required validation mindset

- Public LB is a **sanity check only**
- Prefer well-level OOF and hidden-interval diagnostics over leaderboard feedback
- If OOF improves but relies on fragile tricks, do not trust it

### Required checks for V6-family work

When changing the V6 line, prioritize these analyses:

1. **Adversarial validation**
   - Use `scripts/adversarial_validation.py`
   - Inspect whether train/test separation is dominated by coordinate or trajectory artifacts

2. **Per-well residual diagnostics**
   - Identify worst wells
   - Inspect long-tail intervals and extrapolation-heavy regions
   - Compare error stability across folds

3. **Feature stability**
   - Prefer features whose usefulness is stable across seeds/folds
   - Be skeptical of highly unstable features that produce isolated gains

4. **CV realism**
   - Prefer shuffled well grouping over sorted GroupKFold
   - Avoid row-level random CV

## File Editing Rules

- Prefer additive notebook evolution; do not overwrite historical benchmark notebooks unless explicitly asked
- Keep Kaggle notebooks self-contained and rerunnable
- Notebooks must read from `/kaggle/input/...` and write `/kaggle/working/submission.csv`
- Do not make the primary notebook depend on local-only artifacts

## Repo Scripts

Useful scripts currently in this repo:

- `scripts/_build_v5_notebook.py` — builder for v5 notebook
- `scripts/_build_v6_notebook.py` — builder for v6 notebook (replaces `patch_v5_to_v6.py`)
- `scripts/_build_v7_notebook.py` — builder for v7 notebook (leakage fixes + DTW)
- `scripts/_build_v8_notebook.py` — builder for v8 notebook (log-driven robustness)
- `scripts/adversarial_validation.py` — train/test drift diagnostic
- `scripts/train_baseline.py` — conservative baseline training pipeline
- `scripts/make_submission.py` — baseline submission generation

## Documentation Expectations

When important notebook workflows change:

- Update `README.md`
- Update `AGENTS.md`
- Keep this `CLAUDE.md` aligned with the active modeling line

## Practical Rule of Thumb

If a change mainly improves the public leaderboard but increases:
- leakage risk,
- coordinate memorization,
- OOF tuning complexity,
- or Kaggle runtime/memory pressure,

then it is probably the wrong default for this repository.