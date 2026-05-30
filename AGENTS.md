# Repository Agent Policy

## Competition Summary

- Competition: ROGII Wellbore Geology Prediction.
- Kaggle links:
  - Overview: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview
  - Data: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data
  - Rules: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules
- Local evidence: `data/AI_wellbore_geology_prediction_task_en.pdf` and `data/sample_submission.csv`.
- Task goal: predict true vertical thickness for the hidden interval of horizontal wells.
- Target: `TVT` in train horizontal-well files.
- Evaluation metric: RMSE, based on the local task PDF.
- Submission format: `id,tvt`, matching `data/sample_submission.csv`.

## Data Rules

- Preserve original Kaggle files under `data/`.
- Do not edit, overwrite, normalize in place, or regenerate files in `data/train`, `data/test`, or `data/sample_submission.csv`.
- Write generated artifacts only under `outputs/`.
- Do not commit raw archives, large generated intermediates, model binaries, or cache folders unless explicitly requested.
- Local layout currently expected:
  - `data/train/*__horizontal_well.csv`
  - `data/train/*__typewell.csv`
  - `data/train/*.png`
  - `data/test/*__horizontal_well.csv`
  - `data/test/*__typewell.csv`
  - `data/sample_submission.csv`

## Competition Rules Summary

Only facts confirmed from local files or universally safe competition practice should be documented as confirmed. The Kaggle Rules page was not available as usable command-line text during setup, so do not invent rule details.

Items requiring manual confirmation in a browser:

- External-data allowance and restrictions.
- Team merger and collaboration rules.
- Private sharing and code sharing rules.
- Daily and final submission limits.
- Any license, prize, or eligibility language.

## Kaggle Code Requirements (Hard Constraints)

These constraints come from the competition's Code Requirements page and bind every submitted notebook. The Submit button is disabled until all are satisfied:

- **CPU Notebook ≤ 9 hours run-time.**
- **GPU Notebook ≤ 9 hours run-time.**
- **Internet access is disabled** at submission run-time.
- Freely and publicly available external data is allowed, including pre-trained models, provided they are attached as Kaggle datasets (not fetched from the network at run-time).
- **Submission file must be named `submission.csv`** and written to `/kaggle/working/submission.csv`.

Enforcement rules for this repo:

- Notebooks must not call `pip install`, `apt-get`, `wget`, `curl`, `urllib.request`, `requests`, or any other network operation at run-time. If a package is required, it must already be present in the Kaggle base image or attached as a dataset.
- A defensive `try/except ImportError` followed by `subprocess.run(["pip","install",...], check=False)` is acceptable only as a no-op safety net for locally-rerun development (it will silently fail on Kaggle without internet); the notebook must run successfully even when that branch is skipped.
- Any notebook whose CV + full-retrain wall-clock approaches 8 hours must document the runtime budget and have a degraded fallback path (smaller fold count, lower iteration cap, or fewer DTW radii) that still produces `submission.csv`.
- Validate the submission file name on the last cell (`assert OUT.name == "submission.csv"`).

## Modeling Rules for This Repo

- Prevent target leakage. Never use `TVT` as a feature.
- Treat `TVT_input` as revealed context, not as a direct supervised feature, unless a validation design explicitly simulates hidden intervals.
- Do not use random row splits as the primary validation result.
- Prefer well-level or group validation keyed by the filename well id.
- Use train/test-compatible tabular features first: trajectory columns, gamma ray, row position, deltas, missingness flags, and per-well rolling features.
- Do not hardcode sample-submission values, leaderboard feedback, test IDs, or private-set patterns.
- Do not change raw data to make a submission pass validation.

## Reproducibility Rules

- Use CLI scripts for repeatable work.
- Keep fixed random seeds in training scripts.
- Write reports and summary tables under `outputs/tables/`.
- Write figures under `outputs/figures/`.
- Write submissions under `outputs/submissions/`.
- Write model artifacts under the configured generated-output model-artifact directory.
- Report commands, output paths, validation metrics, and known risks after each material run.

## Kaggle Notebook Submission Rules

- This competition should use Kaggle Notebook rerun submission for the final Kaggle submission artifact.
- Maintain Kaggle submission deliverables as `.ipynb` notebooks by default unless explicitly requested otherwise.
- A standalone `.py` Kaggle-copyable kernel may be used only as a temporary internal conversion source or when the user explicitly asks for a `.py` deliverable.
- New Kaggle submission deliverables should default to `.ipynb`; do not create a `.py` deliverable as the primary artifact unless explicitly requested.
- Keep the stable v1 kernel unchanged unless the user explicitly asks to modify it.
- `kaggle_kernel_v1_lightgbm.ipynb` is the stable v1 submission notebook.
- `kaggle_kernel_v2_lightgbm.ipynb` is the full experimental v2 stronger notebook and is known to exceed Kaggle memory.
- `kaggle_kernel_v2_lightgbm_fast.ipynb` is the memory-safe experimental v2-fast notebook.
- `kaggle_kernel_v3_residual_lgbm.ipynb` is the previous best residual LightGBM candidate, with Public LB RMSE `14.527`.
- `kaggle_kernel_v3_residual_lgbm.ipynb` must not be overwritten; future candidates should be additive notebooks unless the user explicitly requests otherwise.
- `kaggle_kernel_v4_residual_optimized.ipynb` is the previous best residual optimization candidate, with Public LB RMSE `13.764`; it is a residual-anchor optimization candidate informed by public/reference notebooks.
- `kaggle_kernel_v5_lgb_xgb_residual.ipynb` is the current best Public LB candidate, with Public LB RMSE `10.013`.
- `kaggle_kernel_v6_regularized_robust_ensemble.ipynb` is the v6 robust experimental notebook: preserves v5 residual-anchor, adds shuffled GroupKFold, stronger LGB regularization, CatBoost diversity, relative trajectory features, and clean `alpha * tau` post-processing. Public LB `9.791`.
- `kaggle_kernel_v7_full_retrain.ipynb` is the v7 notebook: adds three leakage fixes over v6 (per-fold KNN rebuild, IDW fallback for `FormationPlaneKNN` extrapolation, per-fold `alpha×tau` with median aggregation), full-data retraining at `mean_best_iter × 1.10`, Nelder-Mead ensemble weights, and Sakoe-Chiba DTW features. CV runtime ≈ 6 hours.
- `kaggle_kernel_v8_robust.ipynb` is the v8 notebook: log-driven fixes over v7 — robust retrain iters (`max(median, p75×0.8) × 1.10`), `build_well` NaN/Inf sanitation, wider pp grid (alpha 0.70–1.15, tau 25–200 with finer resolution), stratified GroupKFold by well length, diversified LGB seeds (different `num_leaves`/`learning_rate`), and LGB early-stopping patience 300. Public LB `10.143`; OOF 10.650; actual runtime 7.78h. Nelder-Mead zeroed all LGB weights (xgb=0.518, cb=0.474, lgb≈0), confirming DTW features hurt LGB and will be removed in v9.
- Do not treat local `outputs/submissions/submission_baseline.csv` as the direct final Kaggle submission source.
- The Kaggle Notebook must generate `/kaggle/working/submission.csv`.
- The standalone notebook must read competition data from Kaggle-mounted `/kaggle/input/`, preferring `/kaggle/input/rogii-wellbore-geology-prediction/` and allowing recursive auto-detection of an attached input directory containing `sample_submission.csv`, `train/`, and `test/`.
- The standalone notebook must not depend on local trained model artifacts.
- Update `README.md` whenever important scripts or submission workflows change.
- Do not repeatedly tune against the Public Leaderboard. Use Public LB only as a notebook-submission sanity check; prefer well/group-level validation and hidden-interval diagnostics for candidate selection.
- The Public LB is calculated on approximately 26% of the test data; final results use the other 74%, so do not continue model changes from Public LB alone.
- Stronger Kaggle kernels must respect Kaggle CPU RAM limits. Avoid broad rolling-window grids, repeated aggregate merges, large OOF arrays, multi-seed ensembles, and feature explosions that materially increase train-matrix memory.
- Stable submitted/reference remains v1. Current best Public LB candidate is v5. v4 remains the previous best residual optimization reference, and v3 remains the earlier residual reference.
- For the primary modeling line, prefer residual GBDT around the last known TVT anchor: train on `TVT - last_known_TVT_input`, then reconstruct submitted `tvt` from the anchor plus predicted residual.
- Future primary modeling work should preserve the residual-anchor framework unless the user explicitly asks for a separate experimental line.
- Before continuing public-notebook-inspired modeling, audit the local LightGBM and XGB public notebook copies and document transferable choices, leakage risks, and memory risks.
- Before adding public-notebook-inspired features, consult `12-049-rogii-eda-leakageriskdiscussion.ipynb` and document why the feature is leakage-safe under either the strict or offline prediction policy.
- Avoid full public-notebook feature explosions unless memory safety has been proven. Compact residual LightGBM features are preferred over copying hundreds of candidate features.
- Public Leaderboard feedback is a sanity check only; do not use it as the sole model-selection criterion.
- For the V6 line and later robust notebooks, require shuffled well-level CV, adversarial-validation awareness, and hidden-interval diagnostics before trusting an apparent OOF gain.
- Avoid OOM-prone feature explosions, broad beam-search grids, large OOF arrays, and heavy multi-model ensembles unless the user explicitly requests that experiment and memory safety is documented.
- Distinguish between spatial priors and leakage: local neighborhood spatial priors may be acceptable, but exact coordinate hard replacement / direct label copy is not.
- Avoid post-processing tricks that add extra OOF tuning dimensions without stable evidence, including Savitzky-Golay smoothing and `w_pf` blend search in the primary notebook line.
- Treat CatBoost as an ensemble-diversity tool first, not a license for heavy hyperparameter search.
- Be skeptical of improvements that are driven mainly by coordinate identity or public-LB-specific overlap behavior.
- Avoid LSTM or other deep-learning kernels unless the user explicitly asks for an experimental sequence notebook.
- Deep-learning submission notebooks must be explicitly memory-safe for Kaggle CPU reruns: cap sequence length, window counts, epochs, batch size, and feature count; avoid constructing full 3D train tensors; and keep dynamic per-batch slicing or similarly bounded data access.
- Deep-learning and other experimental notebooks must keep a reliable fallback that still writes `/kaggle/working/submission.csv` if the experimental model fails, runs out of memory, or produces non-finite predictions.
- Experimental notebooks must not modify or invalidate the stable v1 notebook, and they must not treat Public Leaderboard feedback as the sole optimization target.

## Agent Workflow

- Read this file before changing code, documentation, or outputs.
- Inspect local data schema before EDA, training, or submission generation.
- Update `README.md` after important script or output changes.
- Update `README.md` after every major Kaggle notebook or submission-workflow change.
- Keep changes additive and easy to rerun.
- If a Kaggle rule or competition fact cannot be confirmed locally, mark it as requiring manual confirmation instead of guessing.

## Items Requiring Manual Confirmation

The Kaggle Rules page details listed above still require manual browser confirmation. Until confirmed, this repository should use conservative competition behavior: no external data, no private sharing, no rule-sensitive automation, and no assumptions about submission limits.
