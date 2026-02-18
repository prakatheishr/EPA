import os
from pathlib import Path

import pandas as pd
from PIL import Image

MES_ALLOWED = {"MES-0", "MES-1", "MES-2", "MES-3"}
REQUIRED_COLS = {"img_url", "mes_scoring_0_3", "description"}

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