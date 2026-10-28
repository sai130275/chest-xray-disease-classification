"""
Backwards-compatibility entry point for RAD-X dataset.
Core implementation resides in ml.dataset.
"""

from ml.dataset import (
    ChestXRayDataset,
    load_data,
    get_dataloaders,
    train_transform,
    val_transform,
    CSV_PATH,
    IMAGE_DIR,
)

__all__ = [
    "ChestXRayDataset",
    "load_data",
    "get_dataloaders",
    "train_transform",
    "val_transform",
    "CSV_PATH",
    "IMAGE_DIR",
]

if __name__ == "__main__":
    print(f"CSV Path: {CSV_PATH}")
    print(f"Image Dir: {IMAGE_DIR}")
    train_df, val_df, test_df, mlb = load_data()
    print("Dataset loaded successfully via ml.dataset")
