import os
from pathlib import Path

import pandas as pd
from PIL import Image

MES_ALLOWED = {"MES-0", "MES-1", "MES-2", "MES-3"}
REQUIRED_COLS = {"img_url", "mes_scoring_0_3", "description"}

EXTS = [".jpg", ".jpeg", ".png"]


def resolve_image_path(image_root: Path, img_url: str) -> Path | None:
    p = image_root / img_url
    if p.exists():
        return p

    # Try extension swaps only if it looks like an image we can swap
    name_lower = p.name.lower()
    for s in EXTS:
        if name_lower.endswith(s):
            stem = p.with_suffix("")  # remove extension
            for alt in EXTS:
                alt_p = stem.with_suffix(alt)
                if alt_p.exists():
                    return alt_p

    return None


def main() -> None:
    csv_path = os.environ.get("UC_DATA_CSV")
    image_root = os.environ.get("UC_IMAGE_ROOT")

    if not csv_path or not image_root:
        raise SystemExit(
            "Missing env vars. Set:\n"
            "  export UC_DATA_CSV='/path/to/dataframe_complete.csv'\n"
            "  export UC_IMAGE_ROOT='/path/to/'"
        )

    csv_path = Path(csv_path)
    image_root = Path(image_root)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not image_root.exists():
        raise SystemExit(f"Image root not found: {image_root}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    bad_labels = sorted(set(df["mes_scoring_0_3"].astype(str)) - MES_ALLOWED)
    if bad_labels:
        raise SystemExit(f"Found invalid mes_scoring_0_3 labels: {bad_labels}")

    # Check a sample of files exist + can be opened
    sample = df.sample(min(25, len(df)), random_state=42)
    missing_files = []
    for _, row in sample.iterrows():
        img_url = str(row["img_url"])
        p = resolve_image_path(image_root, img_url)

        if p is None:
            missing_files.append(str(image_root / img_url))
            continue

        try:
            Image.open(p).convert("RGB")
        except Exception as e:
            raise SystemExit(f"Failed to open image: {p}\n{e}")

    if missing_files:
        print("Some files missing (showing up to 10):")
        for p in missing_files[:10]:
            print("  ", p)
        raise SystemExit("Fix UC_IMAGE_ROOT or img_url paths (or missing files).")

    print("Check passed:")
    print(f"- Rows: {len(df)}")
    print(f"- CSV: {csv_path}")
    print(f"- Image root: {image_root}")
    print("- Labels OK, sample images exist")


if __name__ == "__main__":
    main()
