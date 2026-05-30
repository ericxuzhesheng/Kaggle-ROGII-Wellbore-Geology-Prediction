from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SEED = 2026
TARGET = "TVT"
SUBMISSION_TARGET = "tvt"


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = DATA_DIR
    outputs: Path = OUTPUT_DIR
    tables: Path = OUTPUT_DIR / "tables"
    figures: Path = OUTPUT_DIR / "figures"
    models: Path = OUTPUT_DIR / "models"
    oof: Path = OUTPUT_DIR / "oof"
    submissions: Path = OUTPUT_DIR / "submissions"

    def ensure(self) -> None:
        for path in (self.outputs, self.tables, self.figures, self.models, self.oof, self.submissions):
            path.mkdir(parents=True, exist_ok=True)


def well_id_from_path(path: Path) -> str:
    return path.name.split("__", 1)[0].split(".", 1)[0]


def list_files(split: str, kind: str) -> list[Path]:
    folder = DATA_DIR / split
    if kind == "horizontal":
        pattern = "*__horizontal_well.csv"
    elif kind == "typewell":
        pattern = "*__typewell.csv"
    elif kind == "png":
        pattern = "*.png"
    else:
        raise ValueError(f"Unknown file kind: {kind}")
    return sorted(folder.glob(pattern))


def load_horizontal(split: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in list_files(split, "horizontal"):
        df = pd.read_csv(path)
        df.insert(0, "well_id", well_id_from_path(path))
        df.insert(1, "row_id", np.arange(len(df), dtype=np.int32))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No horizontal-well CSV files found for split={split!r}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_typewell_summary(split: str) -> pd.DataFrame:
    rows = []
    for path in list_files(split, "typewell"):
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number])
        row = {"well_id": well_id_from_path(path), "typewell_rows": len(df)}
        for col in numeric.columns:
            row[f"typewell_{col}_mean"] = numeric[col].mean()
            row[f"typewell_{col}_std"] = numeric[col].std()
            row[f"typewell_{col}_min"] = numeric[col].min()
            row[f"typewell_{col}_max"] = numeric[col].max()
        rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_ids(df: pd.DataFrame) -> pd.Series:
    return df["well_id"].astype(str) + "_" + df["row_id"].astype(str)


def missing_tvt_input_mask(df: pd.DataFrame) -> pd.Series:
    if "TVT_input" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["TVT_input"].isna()


def add_features(df: pd.DataFrame, use_typewell: bool = True) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    work = work.sort_values(["well_id", "row_id"]).reset_index(drop=True)

    if use_typewell:
        split = "test" if TARGET not in work.columns else "train"
        type_summary = load_typewell_summary(split)
        if not type_summary.empty:
            work = work.merge(type_summary, on="well_id", how="left")

    base_numeric = [
        col
        for col in ["MD", "X", "Y", "Z", "GR"]
        if col in work.columns
    ]

    for col in base_numeric:
        work[f"{col}_missing"] = work[col].isna().astype(np.int8)
        work[col] = work.groupby("well_id", sort=False)[col].transform(lambda s: s.ffill().bfill())
        work[col] = work[col].fillna(work[col].median())
        work[f"{col}_from_start"] = work[col] - work.groupby("well_id", sort=False)[col].transform("first")
        work[f"{col}_diff1"] = work.groupby("well_id", sort=False)[col].diff().fillna(0.0)

    size = work.groupby("well_id", sort=False)["row_id"].transform("max").replace(0, 1)
    work["row_frac"] = work["row_id"] / size
    work["rows_in_well"] = size + 1

    if "TVT_input" in work.columns:
        work["tvt_input_missing"] = work["TVT_input"].isna().astype(np.int8)
        known_md = work["MD"].where(work["TVT_input"].notna()) if "MD" in work.columns else pd.Series(np.nan, index=work.index)
        last_known_md = known_md.groupby(work["well_id"], sort=False).ffill()
        work["md_since_last_tvt_input"] = (work["MD"] - last_known_md).fillna(0.0) if "MD" in work.columns else 0.0
    else:
        work["tvt_input_missing"] = 0
        work["md_since_last_tvt_input"] = 0.0

    if "GR" in work.columns:
        group = work.groupby("well_id", sort=False)["GR"]
        work["GR_lag1"] = group.shift(1)
        work["GR_lag3"] = group.shift(3)
        work["GR_roll5_mean"] = group.transform(lambda s: s.rolling(5, min_periods=1).mean())
        work["GR_roll15_mean"] = group.transform(lambda s: s.rolling(15, min_periods=1).mean())
        work["GR_roll15_std"] = group.transform(lambda s: s.rolling(15, min_periods=2).std())
        for col in ["GR_lag1", "GR_lag3", "GR_roll15_std"]:
            work[col] = work.groupby("well_id", sort=False)[col].transform(lambda s: s.bfill().ffill()).fillna(work["GR"].median())

    allowed_raw = {"row_id", "MD", "X", "Y", "Z", "GR"}
    allowed_generated_prefixes = (
        "typewell_",
        "GR_",
    )
    allowed_generated_suffixes = (
        "_missing",
        "_from_start",
        "_diff1",
    )
    allowed_named = {
        "row_frac",
        "rows_in_well",
        "tvt_input_missing",
        "md_since_last_tvt_input",
    }
    exclude = {"well_id", TARGET, "TVT_input"}
    feature_cols = [
        col
        for col in work.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(work[col])
        and (
            col in allowed_raw
            or col in allowed_named
            or col.startswith(allowed_generated_prefixes)
            or col.endswith(allowed_generated_suffixes)
        )
    ]
    work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = work[feature_cols].median(numeric_only=True).fillna(0.0)
    work[feature_cols] = work[feature_cols].fillna(medians)
    return work, feature_cols


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def validate_submission(path: Path) -> dict[str, object]:
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = pd.read_csv(path)
    problems: list[str] = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"columns {list(sub.columns)} do not match {list(sample.columns)}")
    if len(sub) != len(sample):
        problems.append(f"row count {len(sub)} does not match {len(sample)}")
    if "id" in sub.columns and not sub["id"].equals(sample["id"]):
        problems.append("id order does not match sample_submission.csv")
    if SUBMISSION_TARGET in sub.columns:
        values = sub[SUBMISSION_TARGET].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            problems.append("submission contains NaN or infinite predictions")
    return {"ok": not problems, "problems": problems, "rows": len(sub)}
