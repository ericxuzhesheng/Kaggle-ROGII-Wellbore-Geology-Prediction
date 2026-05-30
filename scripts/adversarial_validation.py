"""Adversarial validation script for finding target-leakage features in ROGII.

This trains a classifier to distinguish train vs test split. High feature
importances indicate distribution shift or memorization of split-specific
artifacts.

Features with AUC > 0.6 are suspicious; features that dominate the importance
should be inspected closely and potentially removed from the final model to
avoid hidden-LB collapse.
"""
from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def get_paths():
    repo = Path(__file__).resolve().parents[1]
    data = repo / "data"
    out = repo / "outputs" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    return data, out


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test horizontal wells."""
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    def _load_dir(d_path):
        paths = sorted(d_path.glob("*__horizontal_well.csv"))
        dfs = []
        for p in paths:
            df = pd.read_csv(p)
            df["well_id"] = p.stem.replace("__horizontal_well", "")
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    logging.info("Loading raw data...")
    train_df = _load_dir(train_dir)
    test_df = _load_dir(test_dir)
    return train_df, test_df


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a standard subset of features for the adversarial test."""
    logging.info("Building basic features...")
    # Add a row_id to keep things ordered
    df = df.copy()
    df["row_id"] = np.arange(len(df))

    # We only care about the evaluation points (where TVT_input is NaN)
    # since that's what the model will predict on.
    # However, to avoid train/test size imbalance driving the classifier,
    # we take the tail of train wells where TVT_input is NaN, AND
    # some known rows to see if coordinate distributions differ.

    ev = df[df["TVT_input"].isna()].copy()

    # Basic features that might be leaky
    features = {}
    features["X"] = ev["X"].values
    features["Y"] = ev["Y"].values
    features["Z"] = ev["Z"].values
    features["MD"] = ev["MD"].values
    features["GR"] = ev["GR"].values

    # Prefix baselines (these simulate the "known" state before evaluation)
    kn = df[df["TVT_input"].notna()].groupby("well_id").last().reset_index()
    kn_map = kn.set_index("well_id")

    ev_kn = ev["well_id"].map(kn_map.to_dict("index"))
    features["last_known_X"] = [x["X"] for x in ev_kn]
    features["last_known_Y"] = [x["Y"] for x in ev_kn]
    features["last_known_Z"] = [x["Z"] for x in ev_kn]
    features["last_known_MD"] = [x["MD"] for x in ev_kn]

    # Relative features (should be safer than absolute)
    features["dx"] = features["X"] - features["last_known_X"]
    features["dy"] = features["Y"] - features["last_known_Y"]
    features["dz"] = features["Z"] - features["last_known_Z"]
    features["md_since"] = features["MD"] - features["last_known_MD"]

    out = pd.DataFrame(features)
    out["well_id"] = ev["well_id"].values
    return out


def run_adversarial_validation():
    data_dir, out_dir = get_paths()
    train_raw, test_raw = load_raw_data(data_dir)

    # Build features
    X_train = build_basic_features(train_raw)
    X_test = build_basic_features(test_raw)

    # Label: 0 for train, 1 for test
    X_train["is_test"] = 0
    X_test["is_test"] = 1

    # Combine
    df = pd.concat([X_train, X_test], ignore_index=True)

    # Exclude non-features
    exclude = ["is_test", "well_id"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols]
    y = df["is_test"]

    logging.info(f"Training adversarial classifier on {len(feature_cols)} features, {len(df)} rows")

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(df))
    importances = np.zeros(len(feature_cols))

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": 42
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        oof_preds[val_idx] = model.predict(X_va)
        importances += model.feature_importance(importance_type="gain") / 5.0

        fold_auc = roc_auc_score(y_va, oof_preds[val_idx])
        logging.info(f"Fold {fold} AUC: {fold_auc:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    logging.info(f"Overall Adversarial AUC: {overall_auc:.4f}")

    if overall_auc > 0.6:
        logging.warning("⚠️ High adversarial AUC! Train and test distributions differ significantly.")
    else:
        logging.info("✅ Train and test distributions are similar.")

    # Save importances
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    logging.info("\nTop 10 Leakage-Prone Features:")
    logging.info(imp_df.head(10).to_string())

    imp_df.to_csv(out_dir / "adversarial_importances.csv", index=False)

    with open(out_dir / "adversarial_summary.json", "w") as f:
        json.dump({"overall_auc": overall_auc}, f, indent=2)


if __name__ == "__main__":
    run_adversarial_validation()
