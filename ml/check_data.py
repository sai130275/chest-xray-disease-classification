"""
RAD-X Data Verification Utility
Verifies dataset integrity, image count, finding label frequencies,
and confirms that all referenced images exist on disk.
"""

import os
from pathlib import Path
import pandas as pd

try:
    from ml.config import CSV_PATH, IMAGE_DIR
except ImportError:
    from config import CSV_PATH, IMAGE_DIR

def run_data_check():
    print(f"Checking dataset CSV: {CSV_PATH}")
    print(f"Checking image directory: {IMAGE_DIR}")

    if not Path(CSV_PATH).exists():
        print(f"ERROR: CSV file not found at {CSV_PATH}")
        return False

    if not Path(IMAGE_DIR).is_dir():
        print(f"ERROR: Image directory not found at {IMAGE_DIR}")
        return False

    df = pd.read_csv(CSV_PATH)
    print(f"Total Rows: {len(df)}")
    print(f"Total Unique Patients: {df['Patient ID'].nunique()}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTop 15 Label Combinations:")
    print(df["Finding Labels"].value_counts().head(15))

    images_on_disk = set(os.listdir(IMAGE_DIR))
    print(f"\nImages found in folder: {len(images_on_disk)}")

    missing = [
        x for x in df["Image Index"]
        if str(x) not in images_on_disk
    ]

    print(f"Missing images referenced in CSV: {len(missing)}")
    if missing:
        print(f"Sample missing: {missing[:5]}")
        return False

    print("\nData integrity check: PASSED successfully.")
    return True

if __name__ == "__main__":
    run_data_check()
