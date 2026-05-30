from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from src.data_utils import SEED, TARGET, Paths, add_features, load_horizontal, rmse


def build_model():
    try:
        from lightgbm import LGBMRegressor

        return "lightgbm", LGBMRegressor(
            objective="regression",
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        try:
            from xgboost import XGBRegressor

            return "xgboost", XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=SEED,
                n_jobs=-1,
                tree_method="hist",
            )
        except Exception:
            return "hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, random_state=SEED)


def main() -> None:
    paths = Paths()
    paths.ensure()
    raw = load_horizontal("train")
    raw = raw.loc[raw[TARGET].notna()].copy()
    data, feature_cols = add_features(raw)
    groups = data["well_id"].to_numpy()
    y = data[TARGET].to_numpy(dtype=float)
    X = data[feature_cols]

    n_splits = min(5, pd.Series(groups).nunique())
    cv = GroupKFold(n_splits=n_splits)

    oof_mean = np.zeros(len(data), dtype=float)
    oof_model = np.zeros(len(data), dtype=float)
    rows = []
    model_name, _ = build_model()

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X.iloc[train_idx], y[train_idx])
        pred_mean = dummy.predict(X.iloc[valid_idx])
        oof_mean[valid_idx] = pred_mean

        _, model = build_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        pred_model = model.predict(X.iloc[valid_idx])
        oof_model[valid_idx] = pred_model

        rows.append(
            {
                "fold": fold,
                "valid_wells": int(pd.Series(groups[valid_idx]).nunique()),
                "valid_rows": len(valid_idx),
                "dummy_mean_rmse": rmse(y[valid_idx], pred_mean),
                f"{model_name}_rmse": rmse(y[valid_idx], pred_model),
            }
        )

    rows.append(
        {
            "fold": "overall",
            "valid_wells": int(pd.Series(groups).nunique()),
            "valid_rows": len(data),
            "dummy_mean_rmse": rmse(y, oof_mean),
            f"{model_name}_rmse": rmse(y, oof_model),
        }
    )
    cv_results = pd.DataFrame(rows)
    cv_results.to_csv(paths.tables / "baseline_cv_results.csv", index=False)

    oof = data[["well_id", "row_id", TARGET]].copy()
    oof["pred_dummy_mean"] = oof_mean
    oof[f"pred_{model_name}"] = oof_model
    oof.to_csv(paths.oof / "baseline_oof_predictions.csv", index=False)

    _, final_model = build_model()
    final_model.fit(X, y)
    artifact = {"model_name": model_name, "model": final_model, "feature_cols": feature_cols, "train_rows": len(data)}
    joblib.dump(artifact, paths.models / "baseline_model.joblib")

    if hasattr(final_model, "feature_importances_"):
        importance = pd.DataFrame({"feature": feature_cols, "importance": final_model.feature_importances_}).sort_values("importance", ascending=False)
    elif isinstance(final_model, RandomForestRegressor):
        importance = pd.DataFrame({"feature": feature_cols, "importance": final_model.feature_importances_}).sort_values("importance", ascending=False)
    else:
        importance = pd.DataFrame({"feature": feature_cols, "importance": np.nan})
    importance.to_csv(paths.tables / "baseline_feature_importance.csv", index=False)

    print(cv_results.tail(1).to_string(index=False))
    print(f"Wrote {paths.models / 'baseline_model.joblib'}")


if __name__ == "__main__":
    main()
