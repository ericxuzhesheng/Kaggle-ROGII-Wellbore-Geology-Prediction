from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data_utils import TARGET, Paths, load_horizontal, missing_tvt_input_mask


def main() -> None:
    paths = Paths()
    paths.ensure()
    train = load_horizontal("train")
    test = load_horizontal("test")

    group_summary = (
        train.groupby("well_id")
        .agg(rows=("row_id", "size"), tvt_min=(TARGET, "min"), tvt_max=(TARGET, "max"), gr_mean=("GR", "mean"), gr_std=("GR", "std"))
        .reset_index()
    )
    group_summary.to_csv(paths.tables / "train_group_summary.csv", index=False)

    target_summary = train[TARGET].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("TVT").reset_index()
    target_summary.to_csv(paths.tables / "target_summary.csv", index=False)

    missing_summary = []
    for split, df in [("train", train), ("test", test)]:
        for col in df.columns:
            missing_summary.append({"split": split, "column": col, "missing": int(df[col].isna().sum()), "missing_rate": float(df[col].isna().mean())})
    pd.DataFrame(missing_summary).to_csv(paths.tables / "missingness_summary.csv", index=False)

    hidden = test.loc[missing_tvt_input_mask(test)].copy()
    hidden["id"] = hidden["well_id"].astype(str) + "_" + hidden["row_id"].astype(str)
    hidden[["id", "well_id", "row_id", "MD", "GR"]].to_csv(paths.tables / "test_hidden_rows.csv", index=False)

    plt.figure(figsize=(8, 5))
    train[TARGET].hist(bins=60)
    plt.xlabel("TVT")
    plt.ylabel("Rows")
    plt.title("Train TVT Distribution")
    plt.tight_layout()
    plt.savefig(paths.figures / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    group_summary["rows"].hist(bins=40)
    plt.xlabel("Rows per well")
    plt.ylabel("Wells")
    plt.title("Train Well Row Counts")
    plt.tight_layout()
    plt.savefig(paths.figures / "well_row_counts.png", dpi=150)
    plt.close()

    print(f"Wrote EDA tables under {paths.tables}")
    print(f"Wrote EDA figures under {paths.figures}")


if __name__ == "__main__":
    main()
