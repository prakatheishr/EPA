import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    csv_path = os.environ.get("UC_DATA_CSV")
    if not csv_path:
        raise SystemExit("Set UC_DATA_CSV env var first.")

    df = pd.read_csv(csv_path)

    if "mes_scoring_0_3" not in df.columns:
        raise SystemExit("CSV missing mes_scoring_0_3 column")

    # Stratified split: 70/15/15
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["mes_scoring_0_3"].astype(str),
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["mes_scoring_0_3"].astype(str),
    )

    out_dir = Path("splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Wrote splits:")
    print(f"- {train_path} ({len(train_df)})")
    print(f"- {val_path} ({len(val_df)})")
    print(f"- {test_path} ({len(test_df)})")
    print("\nClass distribution (train):")
    print(train_df["mes_scoring_0_3"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
