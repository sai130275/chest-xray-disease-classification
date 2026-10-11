import pandas as pd
import os

CSV_PATH = '/Users/prathapagirisaisathwik/Downloads/College/code/ML project/Chest-xray/Project/Chest X Ray.csv'
IMAGE_DIR = '/Users/prathapagirisaisathwik/Downloads/College/code/ML project/Chest-xray/Project/images'

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nLabels:")
print(df["Finding Labels"].value_counts().head(20))

print("\nImages found:", len(os.listdir(IMAGE_DIR)))

missing = [
    x for x in df["Image Index"]
    if not os.path.exists(os.path.join(IMAGE_DIR, x))
]

print("Missing images:", len(missing))