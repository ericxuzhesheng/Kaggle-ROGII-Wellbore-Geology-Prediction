from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.data_utils import DATA_DIR, SUBMISSION_TARGET, Paths, add_features, build_prediction_ids, load_horizontal, missing_tvt_input_mask, validate_submission


def main() -> None:
    paths = Paths()
    paths.ensure()
    artifact_path = paths.models / "baseline_model.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {artifact_path}. Run scripts/train_baseline.py first.")

    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    raw_test = load_horizontal("test")
    data, available_cols = add_features(raw_test)
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Test feature matrix is missing columns used in training: {missing_cols}")
    hidden = data.loc[missing_tvt_input_mask(data)].copy()
    hidden["id"] = build_prediction_ids(hidden)

    predictions = model.predict(hidden[feature_cols])
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    pred = pd.DataFrame({"id": hidden["id"], SUBMISSION_TARGET: predictions})
    sub = sample[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    if sub[SUBMISSION_TARGET].isna().any():
        missing_ids = sub.loc[sub[SUBMISSION_TARGET].isna(), "id"].head().tolist()
        raise ValueError(f"Could not generate predictions for all sample IDs. First missing IDs: {missing_ids}")

    output_path = paths.submissions / "submission_baseline.csv"
    sub.to_csv(output_path, index=False)
    check = validate_submission(output_path)
    if not check["ok"]:
        raise ValueError(f"Submission validation failed: {check['problems']}")

    print(f"Wrote {output_path}")
    print(f"Rows: {check['rows']}")


if __name__ == "__main__":
    main()
