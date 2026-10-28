"""
RAD-X Dataset Module
Handles patient-level stratified data loading, multi-label binarization,
image transformation, and PyTorch DataLoader generation.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from ml.config import CSV_PATH, IMAGE_DIR, NORM_MEAN, NORM_STD, IMAGE_SIZE
except ImportError:
    from config import CSV_PATH, IMAGE_DIR, NORM_MEAN, NORM_STD, IMAGE_SIZE


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for Chest X-Ray Multi-Label Disease Classification.
    Loads images on-demand to minimize memory footprint.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        mlb: MultiLabelBinarizer,
        transform: Optional[transforms.Compose] = None,
        image_dir: Optional[Path | str] = None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.mlb = mlb
        self.transform = transform
        self.image_dir = Path(image_dir) if image_dir else Path(IMAGE_DIR)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.image_dir / str(row["Image Index"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Multi-label binarization
        labels_list = row["Finding Labels"].split("|")
        binary_labels = self.mlb.transform([labels_list])[0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(binary_labels, dtype=torch.float32)


# Data Augmentation & Normalization pipelines
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


def load_data(
    csv_path: Optional[Path | str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MultiLabelBinarizer]:
    """
    Loads dataset metadata, fits MultiLabelBinarizer, and performs
    patient-level train/validation/test split (70% / 15% / 15%)
    strictly avoiding patient identity leakage across splits.
    """
    target_csv = Path(csv_path) if csv_path else Path(CSV_PATH)
    if not target_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {target_csv}")

    df = pd.read_csv(target_csv)
    df["labels_list"] = df["Finding Labels"].str.split("|")

    mlb = MultiLabelBinarizer()
    mlb.fit(df["labels_list"])

    # Split unique patients: 70% train, 30% temp
    unique_patients = df["Patient ID"].unique()

    train_patients, temp_patients = train_test_split(
        unique_patients,
        test_size=0.30,
        random_state=random_state
    )

    # Split temporary patients equally into validation and test (15% / 15%)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=random_state
    )

    train_df = df[df["Patient ID"].isin(train_patients)].copy()
    val_df = df[df["Patient ID"].isin(val_patients)].copy()
    test_df = df[df["Patient ID"].isin(test_patients)].copy()

    # Verify no patient overlap across any splits
    train_ids = set(train_df["Patient ID"])
    val_ids = set(val_df["Patient ID"])
    test_ids = set(test_df["Patient ID"])

    assert train_ids.isdisjoint(val_ids), "Patient overlap detected between train and validation sets!"
    assert train_ids.isdisjoint(test_ids), "Patient overlap detected between train and test sets!"
    assert val_ids.isdisjoint(test_ids), "Patient overlap detected between validation and test sets!"

    print(f"Train: {len(train_df)} images, {len(train_ids)} patients")
    print(f"Validation: {len(val_df)} images, {len(val_ids)} patients")
    print(f"Test: {len(test_df)} images, {len(test_ids)} patients")
    print("Patient overlap check: PASSED")

    return train_df, val_df, test_df, mlb


def get_dataloaders(
    batch_size: int = 32,
    csv_path: Optional[Path | str] = None,
    image_dir: Optional[Path | str] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, MultiLabelBinarizer]:
    """
    Constructs PyTorch DataLoaders for train, val, and test splits.
    """
    train_df, val_df, test_df, mlb = load_data(csv_path=csv_path)

    img_dir = Path(image_dir) if image_dir else Path(IMAGE_DIR)

    train_dataset = ChestXRayDataset(train_df, mlb, transform=train_transform, image_dir=img_dir)
    val_dataset = ChestXRayDataset(val_df, mlb, transform=val_transform, image_dir=img_dir)
    test_dataset = ChestXRayDataset(test_df, mlb, transform=val_transform, image_dir=img_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, mlb
