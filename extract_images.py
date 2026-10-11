import zipfile
import pandas as pd
import os

# ===== PATHS =====
ZIP_PATH = '/Users/prathapagirisaisathwik/Downloads/College/code/ML project/Chest-xray/Project/archive.zip'
CSV_PATH = '/Users/prathapagirisaisathwik/Downloads/College/code/ML project/Chest-xray/Project/Chest X Ray.csv'
OUTPUT_DIR = '/Users/prathapagirisaisathwik/Downloads/College/code/ML project/Chest-xray/Project/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read required filenames
df = pd.read_csv(CSV_PATH)
required = set(df["Image Index"].astype(str))

print(f"Required images: {len(required)}")

# Extract only required images
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    files = z.namelist()

    extracted = 0

    for file in files:
        filename = os.path.basename(file)

        if filename in required:
            z.extract(file, "/tmp/chest_extract")

            src = os.path.join("/tmp/chest_extract", file)
            dst = os.path.join(OUTPUT_DIR, filename)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)

            extracted += 1

            if extracted % 500 == 0:
                print(f"Extracted: {extracted}")

print(f"\nDONE: {extracted} images")