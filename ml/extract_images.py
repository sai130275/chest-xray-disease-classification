"""
RAD-X Image Extraction Utility
Extracts only the images referenced in Chest X Ray.csv from archive.zip.
"""

import os
from pathlib import Path
import zipfile
import pandas as pd

try:
    from ml.config import PROJECT_ROOT, CSV_PATH, IMAGE_DIR
except ImportError:
    from config import PROJECT_ROOT, CSV_PATH, IMAGE_DIR

ZIP_PATH = Path(os.getenv("RADX_ARCHIVE_PATH", PROJECT_ROOT / "archive.zip"))

def extract_required_images():
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Archive not found at {ZIP_PATH}. Skipping extraction.")
        return

    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}. Cannot determine required images.")
        return

    df = pd.read_csv(csv_path)
    required = set(df["Image Index"].astype(str))
    print(f"Required images from CSV: {len(required)}")

    temp_extract = Path("/tmp/chest_extract")
    temp_extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        files = z.namelist()
        extracted = 0

        for file in files:
            filename = os.path.basename(file)
            if filename in required:
                z.extract(file, str(temp_extract))
                src = temp_extract / file
                dst = output_dir / filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src, dst)
                extracted += 1

                if extracted % 500 == 0:
                    print(f"Extracted: {extracted}")

    print(f"\nDONE: {extracted} images extracted to {output_dir}")

if __name__ == "__main__":
    extract_required_images()
