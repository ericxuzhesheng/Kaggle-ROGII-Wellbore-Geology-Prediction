from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_utils import DATA_DIR, Paths, list_files, load_horizontal


def main() -> None:
    paths = Paths()
    paths.ensure()

    rows = []
    for split in ["train", "test"]:
        for kind in ["horizontal", "typewell", "png"]:
            files = list_files(split, kind)
            rows.append(
                {
                    "split": split,
                    "kind": kind,
                    "file_count": len(files),
                    "total_bytes": sum(p.stat().st_size for p in files),
                }
            )
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    rows.append({"split": "sample_submission", "kind": "csv", "file_count": 1, "total_bytes": (DATA_DIR / "sample_submission.csv").stat().st_size})
    pd.DataFrame(rows).to_csv(paths.tables / "data_inventory.csv", index=False)

    summaries = []
    for split in ["train", "test"]:
        df = load_horizontal(split)
        for col in df.columns:
            summaries.append(
                {
                    "split": split,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "rows": len(df),
                    "missing": int(df[col].isna().sum()),
                    "unique": int(df[col].nunique(dropna=True)),
                }
            )
    summaries.append({"split": "sample_submission", "column": "rows", "dtype": "n/a", "rows": len(sample), "missing": 0, "unique": len(sample)})
    pd.DataFrame(summaries).to_csv(paths.tables / "column_summary.csv", index=False)

    print(f"Wrote {paths.tables / 'data_inventory.csv'}")
    print(f"Wrote {paths.tables / 'column_summary.csv'}")


if __name__ == "__main__":
    main()
