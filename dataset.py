
import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torchvision import transforms


CSV_PATH = "/Users/prathapagirisaisathwik/Downloads/College/code/ML project/Chest-xray/Project/Chest X Ray.csv"
IMAGE_DIR = "/Users/prathapagirisaisathwik/Downloads/College/code/ML project/Chest-xray/Project/images"


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, mlb, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.mlb = mlb
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(IMAGE_DIR, row["Image Index"])
        image = Image.open(image_path).convert("RGB")

        labels = self.mlb.transform(
            [row["Finding Labels"].split("|")]
        )[0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["labels_list"] = df["Finding Labels"].str.split("|")

    mlb = MultiLabelBinarizer()
    mlb.fit(df["labels_list"])

    # Split unique patients: 70% train, 30% temporary
    patients = df["Patient ID"].unique()

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=42
    )

    # Split temporary patients equally into validation and test
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=42
    )

    train_df = df[df["Patient ID"].isin(train_patients)].copy()
    val_df = df[df["Patient ID"].isin(val_patients)].copy()
    test_df = df[df["Patient ID"].isin(test_patients)].copy()

    # Verify there is no patient overlap
    train_ids = set(train_df["Patient ID"])
    val_ids = set(val_df["Patient ID"])
    test_ids = set(test_df["Patient ID"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print(
        f"Train: {len(train_df)} images, "
        f"{len(train_ids)} patients"
    )
    print(
        f"Validation: {len(val_df)} images, "
        f"{len(val_ids)} patients"
    )
    print(
        f"Test: {len(test_df)} images, "
        f"{len(test_ids)} patients"
    )
    print("Patient overlap check: PASSED")

    return train_df, val_df, test_df, mlb


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_dataloaders(batch_size=32):
    train_df, val_df, test_df, mlb = load_data()

    train_dataset = ChestXRayDataset(
        train_df, mlb, train_transform
    )
    val_dataset = ChestXRayDataset(
        val_df, mlb, val_transform
    )
    test_dataset = ChestXRayDataset(
        test_df, mlb, val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, mlb
